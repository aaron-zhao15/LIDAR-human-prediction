import os
import datetime
from cfg.test.vel_track_test.vel_track_test_env_config import DigitTestEnvConfig
from cfg.test.vel_track_test.vel_track_test_config import DigitTestConfig
from garage.experiment.deterministic import set_seed
from envs.digit.test.vel_track_test.vel_track_test_env_flat import DigitTestEnvFlat
import numpy as np
import time

# directories
home_path = os.path.dirname(os.path.realpath(__file__)) + "/.."
log_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
log_dir = os.path.join(home_path, 'logs/record_vel_track/' + log_time)

# config
cfg_env = DigitTestEnvConfig()
cfg_test = DigitTestConfig()
cfg_env.vis_record.snapshot_gap = cfg_test.snapshot.snapshot_gap

set_seed(cfg_env.seed)


env = DigitTestEnvFlat(cfg_env, log_dir)

# rollout
max_time = cfg_env.env.max_time * cfg_env.commands.samples.x_vel.shape[0]
max_num_steps = int(max_time / cfg_env.control.control_dt)

eps_length = 0
prev_obs, eps_info = env.reset()

while eps_length < max_num_steps:
    st_time = time.time()
    a, agent_info = np.zeros(12,), {}
    es = env.step(a)    
    eps_length += 1
    if not es.terminal:
        prev_obs = es.observation

    # reset on terminal or time-out
    if es.last:
        prev_obs, eps_info = env.reset()
        
    end_time = time.time()
    if (end_time - st_time) > env.cfg.control.control_dt:
        print("the simulation looks slower than it actually is")
    if (end_time - st_time) < env.cfg.control.control_dt:
        time.sleep(env.cfg.control.control_dt - (end_time - st_time))
