import numpy as np


def calculate_vlr(init_loss, loss, beta):
    w = 1 / (1 + np.exp(-beta * (init_loss - loss) / (loss+1e-6)))
    return w


def calculate_of(train_loss, val_loss, alpha):
    return 1 - np.min([np.power(train_loss / (val_loss + 1e-6), alpha), 1])