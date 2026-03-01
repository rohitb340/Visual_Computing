from monai.losses.dice import DiceFocalLoss
import torch.nn as nn
from pyfed.utils.metric import Metric
from pyfed.loss.loss import *


def build_loss(config):
    if config.TRAIN_LOSS == 'diceloss':
        return DiceLoss()
    if config.TRAIN_LOSS == 'dicece' or config.TRAIN_LOSS == 'joint':
        return JointLoss()
    if config.TRAIN_LOSS == 'tversky':
        return TverskyLoss()
    if config.TRAIN_LOSS == 'tverskyce':
        return TverskyCELoss()
    if config.TRAIN_LOSS == 'FIM':
        return FIMLoss()
    if config.TRAIN_LOSS == 'diceFIM':
        return DiceFIMLoss(lambda_fim=config.L_FIM)
    if config.TRAIN_LOSS == 'tverskyFIM':
        return TverskyFIMLoss()
    if config.TRAIN_LOSS == 'diceFIMce':
        if config.FIM_METHOD is not None:
            return DiceFIMCELoss(lambda_fim=config.L_FIM, method=config.FIM_METHOD)
        else:
            return DiceFIMCELoss(lambda_fim=config.L_FIM)
    if config.TRAIN_LOSS == 'tverskyFIMce':
        return TverskyFIMCELoss()
    if config.TRAIN_LOSS == 'Hausdorff':
        return HausdorffLoss()
    if config.TRAIN_LOSS == 'ce':
        return nn.CrossEntropyLoss()
    if config.TRAIN_LOSS == 'dicefocal':
        return DiceFocalLoss(include_background=True, to_onehot_y=False, softmax=True)
    if config.TRAIN_LOSS == 'edl_dice':
        return EDL_Dice_Loss()
    if config.TRAIN_LOSS == 'edl_dicefundus':
        return EDL_Dice_LossFundus()
    if config.TRAIN_LOSS == 'dicefundus':
        return DiceLossFundus()
    if config.TRAIN_LOSS == 'diceFIMfundus':
        if config.FIM_METHOD is not None:
            return DiceFIMLossFundus(lambda_fim=config.L_FIM, fim_method=config.FIM_METHOD)
        else:
            return DiceFIMLossFundus(lambda_fim=config.L_FIM)
    else:
        raise ValueError('Unknown loss function: {}'.format(config.TRAIN_LOSS))


def build_metric(config):
    return Metric(config.METRIC)

