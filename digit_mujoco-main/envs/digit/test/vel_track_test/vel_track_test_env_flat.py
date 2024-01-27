import numpy as np

from utils.reward_functions import *
from envs.digit import DigitEnvFlat

class DigitTestEnvFlat(DigitEnvFlat):
    def __init__(self, cfg, log_dir=""):
        super().__init__(cfg, log_dir)
        self._reset_counter = 0

    def reset(self):
        ret_val = super().reset()
        self._reset_counter += 1
        return ret_val
    
    def _sample_commands(self):
        """ 
        sample command for test env.
        This should follow the fixed velocity schedule.
        make sure to call mbc.set_usr_command or mbc.reset after this. so that mbc's usr command is sync with env's.
        """
        # Random command sampling in reset
        usr_command = np.zeros(3, dtype=np.float32)
        usr_command[0] = self.cfg.commands.samples.x_vel[self._reset_counter % self.cfg.commands.samples.x_vel.shape[0]]
        usr_command[1] = self.cfg.commands.samples.y_vel[self._reset_counter % self.cfg.commands.samples.y_vel.shape[0]]
        usr_command[2] = self.cfg.commands.samples.ang_vel[self._reset_counter % self.cfg.commands.samples.ang_vel.shape[0]]
        assert self.cfg.commands.resampling_time == 0, "resampling_time must be 0"
        if abs(usr_command[0]) < self.cfg.commands.ranges.cut_off:
            usr_command[0] = 0.
        if abs(usr_command[1]) < self.cfg.commands.ranges.cut_off:
            usr_command[1] = 0.
        if abs(usr_command[2]) < self.cfg.commands.ranges.cut_off:
            usr_command[2] = 0.

        self.usr_command = usr_command