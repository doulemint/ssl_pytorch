from models import build_model
from utils.common_utils import create_config
from optimizer import create_optimizer
from scheduler import create_scheduler, discriminative_lr_params
from dataloader import create_dataloader
from utils.MemoryBank import *
from utils.train_utils import *
from utils.prototype import *
from utils.logger import create_logger
from utils import get_env_info
from losses import create_loss


import argparse


# Parser
parser = argparse.ArgumentParser(description='ssl_base_model')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
args = parser.parse_args()


def main():

    #create config
    config=create_config(args.config_exp)
    output_dir = pathlib.Path(config.train.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    logger = create_logger(name=__name__,
                           distributed_rank=0,
                           output_dir=output_dir,
                           filename='log.txt')
    logger.info(config)
    logger.info(get_env_info(config))
    #hyperparameter
    num_neighbors = config.criterion_kwargs.num_neighbors

    #build model
    # model = get_encoder(config).to(config['device'])
    # model.classifier = nn.Linear(model.classifier.in_features, config.dataset.n_classes)
    model_q = build_model(config,type="query",head="linear").to(config['device']) 

    checkpoint = torch.load(config.train.checkpoint, map_location='cpu')
    if isinstance(model,(nn.DataParallel, nn.parallel.DistributedDataParallel)):
            model.module.load_state_dict(checkpoint['model'])
    else:
            model.load_state_dict(checkpoint['model'])

    labalbed_dataloader = create_dataloader(config,True,True,True)
    val_dataloader = create_dataloader(config,True,False,True)

    memory_bank_val_labeled = MemoryBank(len(labalbed_dataloader)*config.train.batch_size, 
                                config.model.features_dim,
                                config.dataset.n_classes, config.criterion_kwargs.temperature)

    fill_memory_bank(labalbed_dataloader, model, memory_bank_base,config.device)

    memory_bank_val_labeled.cal_mean_rep(config)
    
    acc=memory_bank_val_labeled.get_knn(memory_bank_val_labeled.centroids,memory_bank_val_labeled.features)
    print("knn with centroids acc: ",acc)

    acc=memory_bank_val_labeled.self_clustering_kmeans()
    print("self cluster kmeans acc: ",acc)

    _,acc=memory_bank_val_labeled.mine_nearest_neighbors(3)
    print("mine_nearest_neighbors acc: ",acc)