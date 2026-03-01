from .base import BaseConfig


class Config(BaseConfig):
    def __init__(self, exp_name='fedharmo'):
        super(Config, self).__init__(exp_name)

        self.CLIENT = 'FedHarmoClient'
        self.TRAIN_OPTIMIZER = 'wpadam'
        self.TRAIN_WPADAM_ALPHA = 0.05
        self.COMM_TYPE = 'FedAvg'
        self.AUG_METHOD = None


