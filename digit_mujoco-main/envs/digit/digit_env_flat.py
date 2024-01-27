import datetime
import os

import akro
import numpy as np
import cv2

import transforms3d as tf3
# import collections

from envs.common import robot_interface

from modules import MBCWrapper

import mujoco
import mujoco_viewer

from utils.reward_functions import *
from envs.digit import DigitEnvBase
from gym import utils
import json

from cfg.digit_env_config import DigitEnvConfig

class DigitEnvFlat(DigitEnvBase, utils.EzPickle):
    def __init__(self, cfg, log_dir=""):
        super().__init__(cfg, log_dir)
        assert self.cfg.terrain.terrain_type == 'flat', f"the terrain type should be flat. but got {self.cfg.terrain.terrain_type}"

        # load terrain info     
        self.env_origin = np.zeros(3)
        terrain_dir = os.path.join(self.home_path,'models/'+self.cfg.terrain.terrain_type)

        # load model and data from xml
        path_to_xml_out = os.path.join(terrain_dir, 'digit-v3-'+ self.cfg.terrain.terrain_type+'.xml')
        self.model = mujoco.MjModel.from_xml_path(path_to_xml_out)
        self.data = mujoco.MjData(self.model)
        assert self.model.opt.timestep == self.cfg.env.sim_dt

        # class that have functions to get and set lowlevel mujoco simulation parameters
        self._interface = robot_interface.RobotInterface(self.model, self.data,
                                                         'right-toe-roll', 'left-toe-roll',
                                                         'right-foot', 'left-foot')
        # nominal pos and standing pos
        # self.nominal_qpos = self.data.qpos.ravel().copy() # lets not use this. because nomial pos is weird
        self.nominal_qvel = self.data.qvel.ravel().copy()
        self.nominal_qpos = self.model.keyframe('standing').qpos
        self.nominal_motor_offset = self.nominal_qpos[self._interface.get_motor_qposadr()]

        self._mbc = MBCWrapper(self.cfg, self.nominal_motor_offset, self.cfg.control.action_scale)

        # setup viewer
        self.frames = [] # this only be cleaned at the save_video function
        self._viewer = None
        if self.cfg.vis_record.visualize:
            self.visualize()

        # defualt geom friction
        self.default_geom_friction = self.model.geom_friction.copy()
        # pickling
        kwargs = {"cfg": self.cfg, "log_dir": self.log_dir,}
        utils.EzPickle.__init__(self, **kwargs)
    
    def _reset_state(self):
        init_qpos = self.nominal_qpos.copy()
        init_qvel = self.nominal_qvel.copy()
        init_qpos[0:2] = self.env_origin[:2]

        # dof randomized initialization
        if self.cfg.reset_state.random_dof_reset:
            init_qvel[:6] = init_qvel[:6] + np.random.normal(0, self.cfg.reset_state.root_v_std, 6)
            for joint_name in self.cfg.reset_state.random_dof_names:
                qposadr = self._interface.get_jnt_qposadr_by_name(joint_name)
                qveladr = self._interface.get_jnt_qveladr_by_name(joint_name)                
                init_qpos[qposadr[0]] = init_qpos[qposadr[0]] + np.random.normal(0, self.cfg.reset_state.p_std)                
                init_qvel[qveladr[0]] = init_qvel[qveladr[0]] + np.random.normal(0, self.cfg.reset_state.v_std)

        # # root direction randomize NOTE: don't do this when using mbc as expert. It mess up the target yaw computation
        # init_qpos[3:7] = tf3.quaternions.mat2quat(tf3.euler.euler2mat(0, 0, np.random.uniform(-np.pi, np.pi)))
        self._set_state(
            np.asarray(init_qpos),
            np.asarray(init_qvel)
        )

        # adjust so that no penetration
        rfoot_poses = np.array(self._interface.get_rfoot_keypoint_pos())
        lfoot_poses = np.array(self._interface.get_lfoot_keypoint_pos())
        rfoot_poses = np.array(rfoot_poses)
        lfoot_poses = np.array(lfoot_poses)

        delta = np.max(np.concatenate([0. - rfoot_poses[:, 2], 0. - lfoot_poses[:, 2]]))
        init_qpos[2] = init_qpos[2] + delta + 0.02
        
        self._set_state(
            np.asarray(init_qpos),
            np.asarray(init_qvel)
        )



    def set_command(self, command):
        # this should be used before reset and self.is_command_fixed should be True
        assert self.is_command_fixed
        # assert self._step_cnt is None
        self.usr_command = command
        if self._mbc.model_based_controller is not None:
            self._mbc.set_command(self.usr_command)


    def uploadGPU(self, hfieldid=None, meshid=None, texid=None):
        # hfield
        if hfieldid is not None:
            mujoco.mjr_uploadHField(self.model, self._viewer.ctx, hfieldid)
        # mesh
        if meshid is not None:
            mujoco.mjr_uploadMesh(self.model, self._viewer.ctx, meshid)
        # texture
        if texid is not None:
            mujoco.mjr_uploadTexture(self.model, self._viewer.ctx, texid)

    # def _update_critic_obs(self):
    #     # compute terrains around the robot.
    #     points = self.sample_points_around_robot_root(1.5,9) # (num_points, 2)
    #     points_tmp = np.array(points).transpose() # (2, num_points)
    #     hs = self._get_heights(points_tmp[0], points_tmp[1]) # (num_points,)
    #     assert hs.shape == (81,)
    #     self.value_obs = np.concatenate([self.actor_obs,
    #                                      hs
    #                                      ]).astype(np.float32).flatten()        
        
    #     assert self.value_obs.shape[0] == self.cfg.env.value_obs_dim



    

    
    

if __name__ == "__main__":
    # Create an instance of the class    
    import time
    import numpy as np
    np.random.seed(0)
    
    home_path = os.path.dirname(os.path.realpath(__file__)) + "/.."
    log_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = os.path.join(home_path, 'logs/dagger/' +log_time)
    cfg = DigitEnvConfig()
    env = DigitEnvFlat(cfg, log_dir)
    actor_obs, reset_info = env.reset()    
    while(True):
        time.sleep(0.001)
        es = env.step(np.zeros(12))
        if es.last:
            actor_obs, reset_info = env.reset() 