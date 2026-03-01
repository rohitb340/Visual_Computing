from .base import BaseConfig


class Config(BaseConfig):
    def __init__(self, exp_name='fedevi'):
        super(Config, self).__init__(exp_name)
        self.CLIENT = 'FedEviClient'
        self.COMM_TYPE = 'FedEvi'
        self.TRAIN_LOSS = 'edl_dicefundus'