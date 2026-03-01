from .base import BaseConfig


class Config(BaseConfig):
    def __init__(self, exp_name='fedclam'):
        super(Config, self).__init__(exp_name)
        self.ALPHA = 1
        self.BETA = 1
        self.CLIENT = 'FedCLAMClient'
        self.COMM_TYPE = 'FedCLAM'
        self.ZERO_INIT = False
        self.AGG_LR = 1.0