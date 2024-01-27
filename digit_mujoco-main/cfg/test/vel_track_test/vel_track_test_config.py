from cfg.base.base_config import BaseConfig, ConfigObj

class DigitTestConfig(BaseConfig):
    class architecture(ConfigObj):
        policy_net = [ 512, 256, 64 ]
        value_net = [ 512, 256, 64 ]

    class snapshot(ConfigObj):
        snapshot_gap = 1 # record every time
        snapshot_mode = 'gap'

    class algorithm(ConfigObj):
        gamma = 0.99