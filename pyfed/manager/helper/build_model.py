from pyfed.network.unet import UNet, UNetFedfa

from pyfed.client import (
    BaseClient,
    FedProxClient,
    FedHarmoClient,
    FedBNClient,
    FedSAMClient,
    FedDynClient,
    FedCLAMClient,
    FedEviClient,
)


def build_model(config):
    if config.NETWORK == 'unet':
        model = UNet(**config.NETWORK_PARAMS)
    elif config.NETWORK == 'unetfedfa':
        model = UNetFedfa(**config.NETWORK_PARAMS)
    return model


def build_client(config):
    assert config.CLIENT in [
        'BaseClient',
        'FedProxClient',
        'FedHarmoClient',
        'FedCLAMClient',
        'FedSAMClient',
        'FedDynClient',
        'FedEviClient',
    ]
    client_class = eval(config.CLIENT)

    return client_class