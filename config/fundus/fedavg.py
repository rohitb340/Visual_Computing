from .base import BaseConfig


class Config(BaseConfig):
    def __init__(self, exp_name='fedavg'):
        super(Config, self).__init__(exp_name)

        self.CLIENT = 'BaseClient'
        self.COMM_TYPE = 'FedAvg'
