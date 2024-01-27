import datetime
import os

import akro
import numpy as np
import cv2

import transforms3d as tf3
# import collections

from garage import Environment

import mujoco
import mujoco_viewer

from utils.reward_functions import *
from garage import EnvSpec, EnvStep, StepType
from gym import utils


class DigitEnvBase(Environment):
    def __init__(self, cfg, log_dir=""):
        self.home_path = os.path.dirname(os.path.realpath(__file__)) + "/../.."
        self._env_id = None # this should be set latter in training
        self.log_dir = log_dir

        # config        
        self.cfg = cfg
        
        # constants
        self.max_episode_length = int(self.cfg.env.max_time / self.cfg.control.control_dt)
        self.num_substeps = int(self.cfg.control.control_dt / (self.cfg.env.sim_dt + 1e-8)) + 1
        self.record_interval = int(1 / (self.cfg.control.control_dt * self.cfg.vis_record.record_fps))

        self.history_len = int(self.cfg.env.hist_len_s / self.cfg.control.control_dt)
        self.hist_interval = int(self.cfg.env.hist_interval_s / self.cfg.control.control_dt)      

        self.resampling_time = int(self.cfg.commands.resampling_time / self.cfg.control.control_dt)  

        # control constants that can be changed with DR
        self.kp = self.cfg.control.default_kp
        self.kd = self.cfg.control.default_kd
        self.default_geom_friction = None
        self.motor_joint_friction = np.zeros(20)
        
        # containers (should be reset in reset())    
        self.action = None # only lower body
        self.full_action = None # full body
        self.usr_command = None
        self._step_cnt = None
        self.max_traveled_distance = None

        self.joint_pos_hist = None
        self.joint_vel_hist = None
        # assert self.cfg.env.obs_dim - 70 == int(self.history_len / self.hist_interval) * 12 * 2 # lower motor joints only

        # containers (should be set at the _post_physics_step())        
        self._terminal = None # NOTE: this is internal use only, for the outside terminal check, just use eps.last, eps.terminal, eps.timeout from "EnvStep" or use env_infos "done"
        self.step_type = None

        # containters (should be set at the _get_obs())
        self.actor_obs = None
        self.value_obs = None
        self.robot_state = None

        # containers (should be initialized in child classes)
        self._interface = None
        self.nominal_qvel = None
        self.nominal_qpos = None
        self.nominal_motor_offset = None
        self.model = None
        self.data = None
        self._mbc = None        

        self.curr_terrain_level = None
        # required for Gym and garage compatibility
        self._observation_space = akro.Box(low=-self.cfg.normalization.clip_obs,
                                           high=self.cfg.normalization.clip_obs,
                                           shape=(self.cfg.env.obs_dim,),
                                           dtype=np.float32)
        self._action_space = akro.Box(low=-self.cfg.normalization.clip_act,
                                      high=self.cfg.normalization.clip_act,
                                      shape=(self.cfg.env.act_dim,),
                                      dtype=np.float32)
        self._spec = EnvSpec(action_space=self.action_space,
                             observation_space=self.observation_space,
                             max_episode_length=self.max_episode_length)
        
        
    @property
    def action_space(self):
        """akro.Space: The action space specification."""
        return self._action_space

    @property
    def observation_space(self):
        """akro.Space: The observation space specification."""
        return self._observation_space

    @property
    def spec(self):
        """EnvSpec: The environment specification."""
        return self._spec
    
    def reset(self):
        # domain randomization
        if self.cfg.domain_randomization.is_true:
            self.domain_randomization()
        # TODO: debug everything here
        # reset containers
        self._step_cnt = 0
        self.max_traveled_distance = 0.

        # reset containers that are used in _get_obs
        self.joint_pos_hist = [np.zeros(12)] * self.history_len
        self.joint_vel_hist = [np.zeros(12)] * self.history_len
        self.action = np.zeros(self.cfg.env.act_dim, dtype=np.float32)
        self._sample_commands()

        # setstate for initialization
        self._reset_state()

        # observe for next step
        self._get_obs() # call _reset_state and _sample_commands before this.

        # start rendering              
        if self._viewer is not None and self.cfg.vis_record.visualize:
            frame = self.render()
            if frame is not None:
                self.frames.append(frame)
        
        # reset mbc
        self._do_extra_in_reset() # self.robot_state should be updated before this by calling _get_obs
        
        self._step_assertion_check()

        # second return is for episodic info
        # not sure if copy is needed but to make sure...
        return self.get_eps_info()
    
    def step(self, action):
        if self._step_cnt is None:
            raise RuntimeError('reset() must be called before step()!')
        assert np.all(self._mbc.usr_command == self.usr_command)
        self._mbc_action, self._mbc_torque = self._mbc.get_action(self.robot_state) # this should be called to update phase var and domain
        if self.cfg.control.mbc_control:
            if self.cfg.control.control_type == 'PD':
                action = self._mbc_action.astype(np.float32)
            elif self.cfg.control.control_type == 'T':
                action = self._mbc_torque.astype(np.float32)

        # clip action
        self.action = np.clip(action, self.action_space.low, self.action_space.high)
        self.full_action = np.concatenate((self.action[:6], np.zeros(4), self.action[6:], np.zeros(4))) # action space is only leg. actual motor inlcudes upper body.

        # control step
        if self.cfg.control.control_type == 'PD':
            target_joint = self.full_action * self.cfg.control.action_scale + self.nominal_motor_offset
            if self.cfg.domain_randomization.is_true:
                self.action_delay_time = int(np.random.uniform(0, self.cfg.domain_randomization.action_delay,1) / self.cfg.env.sim_dt)
            self._pd_control(target_joint, np.zeros_like(target_joint))
        if self.cfg.control.control_type == 'T':
            self._torque_control(self.full_action)

        rewards, tot_reward = self._post_physics_step()
        
        return EnvStep(env_spec=self.spec,  # act size, obs size, eps_length
                       action=self.action, # clipped unscaled action
                       reward=tot_reward,
                       observation=self.actor_obs.copy(),  # this observation is next state
                       env_info={
                           'reward_info': rewards,
                           'next_value_obs': self.value_obs.copy(),
                           'curr_value_obs': None,
                           'robot_state': self.robot_state.copy(),
                           'action_label': np.zeros_like(self._mbc_action) if np.isnan(self._mbc_action).any() else self._mbc_action, # this should be given in worker.
                           'terrain_level': self.curr_terrain_level,
                           'done': self.step_type is StepType.TERMINAL or self.step_type is StepType.TIMEOUT,
                           # these are for testing. make sure the command is not resampled in _post_physics_step
                           "tracking_errors":{
                           'x_vel_error': abs(self.root_lin_vel[0] - self.usr_command[0]),
                            'y_vel_error': abs(self.root_lin_vel[1] - self.usr_command[1]),
                            'ang_vel_error': abs(self.root_ang_vel[2] - self.usr_command[2]),
                            "z_vel_error": abs(self.root_lin_vel[2]),
                            "roll_vel_error": abs(self.root_ang_vel[0]),
                            "pitch_vel_error": abs(self.root_ang_vel[1])},
                       },
                       step_type=self.step_type)
    
    def get_eps_info(self):
        """
        return current environment's info.
        These informations are used when starting the episodes. starting obeservations.
        """
        return self.actor_obs.copy(), dict(curr_value_obs=self.value_obs.copy(), robot_state=self.robot_state.copy())
    
    """ 
    internal helper functions 
    """
    def _sample_commands(self):
        """ 
        sample command for env 
        make sure to call mbc.set_usr_command or mbc.reset after this. so that mbc's usr command is sync with env's
        """
        # Random command sampling in reset
        usr_command = np.zeros(3, dtype=np.float32)
        usr_command[0] = np.random.uniform(self.cfg.commands.ranges.x_vel_range[0], 
                                           self.cfg.commands.ranges.x_vel_range[1])
        usr_command[1] = np.random.uniform(self.cfg.commands.ranges.y_vel_range[0], 
                                           self.cfg.commands.ranges.y_vel_range[1])
        usr_command[2] = np.random.uniform(self.cfg.commands.ranges.ang_vel_range[0], 
                                           self.cfg.commands.ranges.ang_vel_range[1])
        if abs(usr_command[0]) < self.cfg.commands.ranges.cut_off:
            usr_command[0] = 0.
        if abs(usr_command[1]) < self.cfg.commands.ranges.cut_off:
            usr_command[1] = 0.
        if abs(usr_command[2]) < self.cfg.commands.ranges.cut_off:
            usr_command[2] = 0.
        self.usr_command = usr_command
        # print("usr_command: ", self.usr_command)

    def _set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

    def _reset_state(self):
        raise NotImplementedError
    
    def _torque_control(self, torque):
        ratio = self._interface.get_gear_ratios().copy()
        for _ in range(self._num_substeps): # this is open loop torque control. no feedback.
            tau = [(i / j) for i, j in zip(torque, ratio)] # TODO: why divide by ratio..? This need to be checked
            self._interface.set_motor_torque(tau)
            self._interface.step()

    def _pd_control(self, target_pos, target_vel):
        self._interface.set_pd_gains(self.kp, self.kd)
        ratio = self._interface.get_gear_ratios().copy()
        for cnt in range(self.num_substeps): # this is PD feedback loop
            if self.cfg.domain_randomization.is_true:
                motor_vel = self._interface.get_act_joint_velocities()
                motor_joint_friction = self.motor_joint_friction * np.sign(motor_vel)
                if cnt < self.action_delay_time:
                    tau = motor_joint_friction
                    tau = [(i / j) for i, j in zip(tau, ratio)]
                    self._interface.set_motor_torque(tau)
                    self._interface.step()
                else:                    
                    tau = self._interface.step_pd(target_pos, target_vel)
                    tau += motor_joint_friction
                    tau = [(i / j) for i, j in zip(tau, ratio)]
                    self._interface.set_motor_torque(tau)
                    self._interface.step()
            else:
                tau = self._interface.step_pd(target_pos, target_vel) # this tau is joint space torque
                tau = [(i / j) for i, j in zip(tau, ratio)]  
                self._interface.set_motor_torque(tau)
                self._interface.step()

    def _post_physics_step(self):
        # update step count and step type. These two shouldn't be used before this call.
        if self.resampling_time != 0 and self._step_cnt % self.resampling_time==0:
            self._sample_commands()
            self._mbc.set_command(self.usr_command)

        # observe for next step
        self._get_obs()
        self._is_terminal()

        # TODO: debug reward function
        rewards, tot_reward = self._compute_reward()

        # visualize
        if self._viewer is not None and self._step_cnt % self.record_interval == 0 and self.cfg.vis_record.visualize:
            frame = self.render()
            if frame is not None:
                self.frames.append(frame)

        self._step_cnt += 1
        self.step_type = StepType.get_step_type(step_cnt=self._step_cnt,
                                           max_episode_length=self.max_episode_length,
                                           done=self._terminal)
        if self.step_type in (StepType.TERMINAL, StepType.TIMEOUT):
            self._step_cnt = None  # this becomes zero when reset is called  

        return rewards, tot_reward
    
    def _get_obs(self):
        """" 
        update actor_obs, value_obs, robot_state, all the other states
        make sure to call _reset_state and _sample_commands before this.
        self._mbc is not reset when first call but it is okay for self._mbc.get_phase_variable(), self._mbc.get_domain(). check those functions.
        """
        # TODO: check all the values
        # update states
        self.qpos = self.data.qpos.copy()
        self.qvel = self.data.qvel.copy()
        self._update_root_state()
        self._update_joint_state()
        self._update_joint_hist()
        self._update_robot_state()
        
        # update observations
        self.projected_gravity = self._interface.get_projected_gravity_vec()
        self.noisy_projected_gravity = self.projected_gravity + np.random.normal(0, self.cfg.obs_noise.projected_gravity_std, 3)
        self.noisy_projected_gravity = self.noisy_projected_gravity / np.linalg.norm(self.noisy_projected_gravity)
        self._update_actor_obs()
        # self.value_obs = self.actor_obs.copy() # TODO: apply dreamwaq
        self._update_critic_obs()

        # update traveled distance
        self.max_traveled_distance = max(self.max_traveled_distance, np.linalg.norm(self.root_xy_pos[:2]))

        # not sure if copy is needed but to make sure...        

    def _update_root_state(self):
        # root states
        self.root_xy_pos = self.qpos[0:2]
        self.root_world_height = self.qpos[2]
        self.root_quat = self.qpos[3:7]
        base_rot = tf3.quaternions.quat2mat(self.root_quat)  # wRb
        self.root_lin_vel = np.transpose(base_rot).dot(self.qvel[0:3])
        self.root_ang_vel = np.transpose(base_rot).dot(self.qvel[3:6])
        self.noisy_root_lin_vel = self.root_lin_vel + np.random.normal(0, self.cfg.obs_noise.lin_vel_std, 3)
        self.noisy_root_ang_vel = self.root_ang_vel + np.random.normal(0, self.cfg.obs_noise.ang_vel_std, 3)
    
    def _update_joint_state(self):
        # motor states
        self.motor_pos = self._interface.get_act_joint_positions()
        self.motor_vel = self._interface.get_act_joint_velocities()
        # passive hinge states
        self.passive_hinge_pos = self._interface.get_passive_hinge_positions()
        self.passive_hinge_vel = self._interface.get_passive_hinge_velocities()

        self.noisy_motor_pos = self.motor_pos + np.random.normal(0, self.cfg.obs_noise.dof_pos_std, 20)
        self.noisy_motor_vel = self.motor_vel + np.random.normal(0, self.cfg.obs_noise.dof_vel_std, 20)
        self.noisy_passive_hinge_pos = self.passive_hinge_pos + np.random.normal(0, self.cfg.obs_noise.dof_pos_std, 10)
        self.noisy_passive_hinge_vel = self.passive_hinge_vel + np.random.normal(0, self.cfg.obs_noise.dof_vel_std, 10)


    def _update_joint_hist(self):
        # joint his buffer update
        self.joint_pos_hist.pop(0)
        self.joint_vel_hist.pop(0)
        if self.cfg.obs_noise.is_true:
            self.joint_pos_hist.append(np.array(self.noisy_motor_pos)[self.cfg.control.lower_motor_index])
            self.joint_vel_hist.append(np.array(self.noisy_motor_vel)[self.cfg.control.lower_motor_index])
        else:
            self.joint_pos_hist.append(np.array(self.motor_pos)[self.cfg.control.lower_motor_index])
            self.joint_vel_hist.append(np.array(self.motor_vel)[self.cfg.control.lower_motor_index])
        assert len(self.joint_vel_hist) == self.history_len

        # assign joint history obs
        self.joint_pos_hist_obs = []
        self.joint_vel_hist_obs = []
        for i in range(int(self.history_len/self.hist_interval)):
            self.joint_pos_hist_obs.append(self.joint_pos_hist[i*self.hist_interval])
            self.joint_vel_hist_obs.append(self.joint_vel_hist[i*self.hist_interval])
        assert len(self.joint_pos_hist_obs) == 3
        self.joint_pos_hist_obs = np.concatenate(self.joint_pos_hist_obs).flatten()
        self.joint_vel_hist_obs = np.concatenate(self.joint_vel_hist_obs).flatten()

    def _update_robot_state(self):
        ''' robot state is state used for MBC '''
        # body_height = self.root_world_height - self._get_height(self.root_xy_pos[0] , self.root_xy_pos[1])
        body_height = self.root_world_height
        root_pos = np.array([self.root_xy_pos[0], self.root_xy_pos[1], body_height])
        self.robot_state = np.concatenate([
            root_pos,  # 2 0~3
            self.root_quat,  # 4 3~7
            self.root_lin_vel,  # 3 7~10
            self.root_ang_vel,  # 3 10~13
            self.motor_pos,  # 20 13~33
            self.passive_hinge_pos,  # 10 33~43
            self.motor_vel,  # 20     43~63
            self.passive_hinge_vel  # 10 63~73
        ])

    def _update_actor_obs(self):
        # NOTE: make sure to call get_action from self._mbc so that phase_variable is updated
        if self.cfg.obs_noise.is_true:
            self.actor_obs = np.concatenate([self.noisy_root_lin_vel * self.cfg.normalization.obs_scales.lin_vel, # 3
                                             self.noisy_root_ang_vel * self.cfg.normalization.obs_scales.ang_vel, # 3
                                             self.noisy_projected_gravity, # 3
                                             self.usr_command * [self.cfg.normalization.obs_scales.lin_vel, 
                                                                 self.cfg.normalization.obs_scales.lin_vel, 
                                                                 self.cfg.normalization.obs_scales.ang_vel], # 3
                                            np.array(self.noisy_motor_pos)[self.cfg.control.lower_motor_index] * self.cfg.normalization.obs_scales.dof_pos, # 12
                                            np.array(self.noisy_passive_hinge_pos) * self.cfg.normalization.obs_scales.dof_pos, # 10
                                            np.array(self.noisy_motor_vel)[self.cfg.control.lower_motor_index] * self.cfg.normalization.obs_scales.dof_vel, # 12
                                            np.array(self.noisy_passive_hinge_vel) * self.cfg.normalization.obs_scales.dof_vel, # 10
                                            np.array([self._mbc.get_phase_variable(), self._mbc.get_domain()]), # 2
                                            self.action.copy(), # 12
                                            self.joint_pos_hist_obs * self.cfg.normalization.obs_scales.dof_pos,
                                            self.joint_vel_hist_obs * self.cfg.normalization.obs_scales.dof_vel]).astype(np.float32).flatten()
        else:
            self.actor_obs = np.concatenate([self.root_lin_vel * self.cfg.normalization.obs_scales.lin_vel, # 3
                                             self.root_ang_vel * self.cfg.normalization.obs_scales.ang_vel, # 3
                                             self.projected_gravity, # 3
                                             self.usr_command * [self.cfg.normalization.obs_scales.lin_vel, 
                                                                 self.cfg.normalization.obs_scales.lin_vel, 
                                                                 self.cfg.normalization.obs_scales.ang_vel], # 3
                                            np.array(self.motor_pos)[self.cfg.control.lower_motor_index] * self.cfg.normalization.obs_scales.dof_pos, # 12
                                            np.array(self.passive_hinge_pos) * self.cfg.normalization.obs_scales.dof_pos, # 10
                                            np.array(self.motor_vel)[self.cfg.control.lower_motor_index] * self.cfg.normalization.obs_scales.dof_vel, # 12
                                            np.array(self.passive_hinge_vel) * self.cfg.normalization.obs_scales.dof_vel, # 10
                                            np.array([self._mbc.get_phase_variable(), self._mbc.get_domain()]), # 2
                                            self.action.copy(), # 12
                                            self.joint_pos_hist_obs * self.cfg.normalization.obs_scales.dof_pos,
                                            self.joint_vel_hist_obs * self.cfg.normalization.obs_scales.dof_vel]).astype(np.float32).flatten()        
        assert self.actor_obs.shape[0] == self.cfg.env.obs_dim
        
    def _update_critic_obs(self):
        self.value_obs = np.concatenate([self.root_lin_vel * self.cfg.normalization.obs_scales.lin_vel, # 3
                                        self.root_ang_vel * self.cfg.normalization.obs_scales.ang_vel, # 3
                                        self.projected_gravity, # 3
                                        self.usr_command * [self.cfg.normalization.obs_scales.lin_vel, 
                                                            self.cfg.normalization.obs_scales.lin_vel, 
                                                            self.cfg.normalization.obs_scales.ang_vel], # 3
                                    np.array(self.motor_pos)[self.cfg.control.lower_motor_index] * self.cfg.normalization.obs_scales.dof_pos, # 12
                                    np.array(self.passive_hinge_pos) * self.cfg.normalization.obs_scales.dof_pos, # 10
                                    np.array(self.motor_vel)[self.cfg.control.lower_motor_index] * self.cfg.normalization.obs_scales.dof_vel, # 12
                                    np.array(self.passive_hinge_vel) * self.cfg.normalization.obs_scales.dof_vel, # 10
                                    np.array([self._mbc.get_phase_variable(), self._mbc.get_domain()]), # 2
                                    self.action.copy(), # 12
                                    self.kp,
                                    self.kd,
                                    self.motor_joint_friction
                                    ]).astype(np.float32).flatten()
        
        assert self.value_obs.shape[0] == self.cfg.env.value_obs_dim
    
    def _do_extra_in_reset(self):
        self._mbc.reset(self.robot_state, self.usr_command)  # get_obs should be called before this to update robot_state

    def _step_assertion_check(self):
        assert self._mbc.get_phase_variable() == 0.
        assert self._mbc.get_domain() == 0
        assert self.usr_command is not None
        assert self._mbc.usr_command is not None
    
    def _is_terminal(self):
        # self_collision_check = self._interface.check_self_collisions()
        # bad_collision_check = self._interface.check_bad_collisions()
        # lean_check = self._interface.check_body_lean()  # TODO: no lean when RL training. why...?
        # terminate_conditions = {"self_collision_check": self_collision_check,
        #                         "bad_collision_check": bad_collision_check,
        #                         # "body_lean_check": lean_check,
        #                         }
        
        root_vel_crazy_check = (self.root_lin_vel[0] > 1.5) or (self.root_lin_vel[1] > 1.5) or (self.root_lin_vel[2] > 1.0) # as in digit controller
        self_collision_check = self._interface.check_self_collisions()
        body_lean_check = self._interface.check_body_lean()
        mbc_divergence_check = np.isnan(self._mbc_torque).any() or np.isnan(self._mbc_action).any() #TODO:remove this when RL.
        terminate_conditions = {"root_vel_crazy_check": root_vel_crazy_check,
                                "self_collision_check": self_collision_check,
                                "body_lean_check": body_lean_check,
                                "mbc_divergence_check": mbc_divergence_check}

        self._terminal = True in terminate_conditions.values()

    def _compute_reward(self):
        # the states are after stepping.
        lin_vel_tracking_reward = lin_vel_tracking(self.root_lin_vel, self.usr_command)        
        ang_vel_tracking_reward = ang_vel_tracking(self.root_ang_vel, self.usr_command)
        z_vel_penalty_reward = z_vel_penalty(self.root_lin_vel)
        roll_pitch_penalty_reward = roll_pitch_penalty(self.root_ang_vel)
        base_orientation_penalty_reward = base_orientation_penalty(self.projected_gravity)
        torque = np.array(self._interface.get_act_joint_torques())[self.cfg.control.lower_motor_index]
        torque_penalty_reward = torque_penalty(torque)

        rfoot_pose = np.array(self._interface.get_rfoot_keypoint_pos()).T
        lfoot_pose = np.array(self._interface.get_lfoot_keypoint_pos()).T
        rfoot_pose = self._interface.change_positions_to_rotated_world_frame(rfoot_pose)
        lfoot_pose = self._interface.change_positions_to_rotated_world_frame(lfoot_pose)

        foot_lateral_distance_penalty_reward = 1.0 if foot_lateral_distance_penalty(rfoot_pose, lfoot_pose) else 0.

        rfoot_grf = self._interface.get_rfoot_grf()
        lfoot_grf = self._interface.get_lfoot_grf()        

        swing_foot_fix_penalty_reward =  swing_foot_fix_penalty(lfoot_grf, rfoot_grf, self.action)

        termination_reward = 1. if self._terminal else 0.

        rewards_tmp = {"lin_vel_tracking": lin_vel_tracking_reward,
                       "ang_vel_tracking": ang_vel_tracking_reward,
                       "z_vel_penalty": z_vel_penalty_reward,
                       "roll_pitch_penalty": roll_pitch_penalty_reward,
                       "base_orientation_penalty": base_orientation_penalty_reward,
                       "torque_penalty": torque_penalty_reward,
                       "foot_lateral_distance_penalty": foot_lateral_distance_penalty_reward,
                       "swing_foot_fix_penalty": swing_foot_fix_penalty_reward,
                       "termination": termination_reward,
                       }        
        rewards = {}
        tot_reward = 0.
        for key in rewards_tmp.keys():
            rewards[key] = getattr(self.cfg.rewards.scales,key) * rewards_tmp[key]
            tot_reward += rewards[key]                

        return rewards, tot_reward

    """
    Visualization Code
    """
    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def visualize(self):
        """Creates a visualization of the environment."""
        assert self.cfg.vis_record.visualize, 'you should set visualize flag to true'
        assert self._viewer is None, 'there is another viewer'
        # if self._viewer is not None:
        #     #     self._viewer.close()
        #     #     self._viewer = None
        #     return
        if self.cfg.vis_record.record:
            self._viewer = mujoco_viewer.MujocoViewer(self.model, self.data, 'offscreen')
        else:
            self._viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer_setup()

    def viewer_setup(self):
        self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self._viewer.cam.fixedcamid = 0
        self._viewer.cam.distance = self.model.stat.extent * 1.5
        self._viewer.cam.lookat[2] = 0.
        self._viewer.cam.lookat[0] = 0
        self._viewer.cam.lookat[1] = 0.
        self._viewer.cam.azimuth = 180
        self._viewer.cam.distance = 5
        self._viewer.cam.elevation = -10
        self._viewer.vopt.geomgroup[0] = 1
        self._viewer._render_every_frame = True
        # self.viewer._run_speed *= 20
        self._viewer._contacts = True
        self._viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = self._viewer._contacts
        self._viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = self._viewer._contacts

    def viewer_is_paused(self):
        return self._viewer._paused
    
    def render(self):       
        assert self._viewer is not None
        if self.cfg.vis_record.record:
            return self._viewer.read_pixels(camid=0)
        else:
            self._viewer.render()
            return None
    
    def save_video(self, name):
        assert self.cfg.vis_record.record
        assert self.log_dir is not None
        assert self._viewer is not None
        video_dir = os.path.join(self.log_dir, "video")
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, name + ".mp4")
        command_path = os.path.join(video_dir, name + ".txt")
        f = open(command_path, "w")
        f.write(str(self.usr_command))
        f.close()
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video_writer = cv2.VideoWriter(video_path, fourcc, self._record_fps,
                                       (self.frames[0].shape[1], self.frames[0].shape[0]))
        for frame in self.frames:
            video_writer.write(frame)
        video_writer.release()
        self.clear_frames()

    def clear_frames(self):
        self.frames = []

    def domain_randomization(self):
        # NOTE: the parameters in mjModel shouldn't be changed in runtime!
        # self.model.geom_friction[:,0] = self.default_geom_friction[:,0] * np.random.uniform(self.cfg.domain_randomization.friction_noise[0],
        #                                                                           self.cfg.domain_randomization.friction_noise[1],
        #                                                                           size=self.default_geom_friction[:,0].shape)
        self.motor_joint_friction = np.random.uniform(self.cfg.domain_randomization.joint_friction[0],
                                                        self.cfg.domain_randomization.joint_friction[1],
                                                        size=self.motor_joint_friction.shape)
        self.kp = self.cfg.control.default_kp * np.random.uniform(self.cfg.domain_randomization.kp_noise[0], self.cfg.domain_randomization.kp_noise[1], size=self.cfg.control.default_kp.shape)
        self.kd = self.cfg.control.default_kd * np.random.uniform(self.cfg.domain_randomization.kd_noise[0], self.cfg.domain_randomization.kd_noise[1], size=self.cfg.control.default_kd.shape)
        # self.model.dof_frictionloss[:] = self._default_frictionloss * np.random.uniform(1-self._domain_noise['joint_friction_noise'][0],
    #                                                                                     1+self._domain_noise['joint_friction_noise'][1],
    #                                                                                     size=self._default_frictionloss.shape)