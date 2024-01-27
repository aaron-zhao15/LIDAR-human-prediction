# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
from cfg.base.base_config import BaseConfig,ConfigObj
from cfg.digit_env_config import DigitEnvConfig

class DigitTestEnvConfig(DigitEnvConfig):
    class env(DigitEnvConfig.env):
        max_time = 5.0  # every this time, the env is reset ans command is updated
    class commands(DigitEnvConfig.commands):
        resampling_time = 0. # commands are updated every reset
        class ranges(DigitEnvConfig.commands.ranges):
            cut_off = 0.0
        class samples(ConfigObj): # this can be adjusted before passed to environment
            x_vel = np.array([0.4])
            y_vel = np.array([0.]) # 0.5
            ang_vel = np.array([0.])

    class control(DigitEnvConfig.control):
        mbc_control = True # if true, mbc action is used in def step()

    class terrain(DigitEnvConfig.terrain): # this can be adjusted before passed to environment
        terrain_type = "flat" # flat, test_rough, rough_random???
        terrain_path = "terrain_info"        
        terrain_level_schedule = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        which_terrain = 0
    
    class vis_record(DigitEnvConfig.vis_record): #TODO: change this
        visualize = True # should set to false when training
        record = False # should visualize true
        record_fps = 15
    
    class domain_randomization(DigitEnvConfig.domain_randomization):
        is_true = False
    
    class obs_noise(DigitEnvConfig.obs_noise):
        is_true = False