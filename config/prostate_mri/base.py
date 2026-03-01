import os
import torch


class BaseConfig:
    def __init__(self, exp_name='fedavg_'):
        self.EXP_NAME = exp_name

        self.NETWORK = 'unet'
        self.NETWORK_PARAMS = {}

        self.DATASET = 'prostate'
        self.INNER_SITES = ['BIDMC', 'HK', 'I2CVB', 'BMC', 'RUNMC', 'UCL']
        self.OUTER_SITES = []
        self.IMAGE_SIZE = [3, 384, 384]

        self.TRAIN_ROUNDS = 500
        self.TRAIN_EPOCH_PER_ROUND = 1
        self.TRAIN_LR = 1e-4
        self.TRAIN_BATCHSIZE = 16
        self.TRAIN_MOMENTUM = 0.9
        self.TRAIN_WEIGHT_DECAY = 1e-4
        self.TRAIN_RESUME = False
        self.TRAIN_AUTO_RESUME = False
        self.TRAIN_GPU = 0
        self.TRAIN_OPTIMIZER = 'adam'
        self.TRAIN_WARMUP_STEPS = 1000
        self.TRAIN_MIN_LR = 1e-5
        self.TRAIN_MODE = 'federated'
        self.TRAIN_LOSS = 'joint'

        self.L_FIM = 1.0
        self.FIM_METHOD = 'w2'
        self.FIM_WARMUP = 10
        self.FIM_RAMPUP = False

        self.TRAIN_RATIO = 0.6
        self.TEST_GPU = 0

        self.METRIC = 'dice'

        self.SEED = 0

        self.COMM_TYPE = 'FedAvg'
        self.CLIENT = 'BaseClient'

        self.__check()

    def __check(self):
        if not torch.cuda.is_available():
            raise ValueError('fedavg_prostate.py: cuda is not avalable')
        if self.DATASET == 'prostate':
            self.DIR_DATA = "pyfed/dataset/Preprocessed_data/"  # PATH TO PROSTATE DATA HERE
            self.DIR_SAVE = os.path.join('results/', self.EXP_NAME)  # PATH TO SAVE EXPERIMENT RESULTS
            self.DIR_CKPT = os.path.join(self.DIR_SAVE, 'ckpt')
            self.DIR_LOG = os.path.join(self.DIR_SAVE, 'log')
        else:
            raise ValueError('Unsupported dataset')

        for path in [self.DIR_SAVE, self.DIR_CKPT]:
            if not os.path.isdir(path):
                os.makedirs(path)

