from config import get_default_config
import torch


def create_config(config_file):
    config = get_default_config()
    if not torch.cuda.is_available():
        config.device = 'cpu'
        config.train.dataloader.pin_memory = False
    else:
        config.device = 'cuda'
    if config_file is not None:
        config.merge_from_file(config_file)
    config.freeze()
    return config