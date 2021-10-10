import yacs,torch
from yacs.config import CfgNode as ConfigNode


def get_env_info(config: yacs.config.CfgNode) -> yacs.config.CfgNode:
    info = {
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda or '',
        'cudnn_version': torch.backends.cudnn.version() or '',
    }
    if config.device != 'cpu':
        info['num_gpus'] = torch.cuda.device_count()
        info['gpu_name'] = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        info['gpu_capability'] = f'{capability[0]}.{capability[1]}'

    return ConfigNode({'env_info': info})