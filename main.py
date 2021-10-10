
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
    model_q = build_model(config,type="query",head="linear").to(config['device'])  
    model_k = build_model(config,type="contra",head="mlp").to(config['device'])  
    #print model info
    # print(model.)


    #setup cudnn
    # setupCuDNN()

    #label dataset
    labalbed_dataloader,unlabeled_dataloader = create_dataloader(config,True,True,False)
    val_dataloader = create_dataloader(config,True,False,True)

    #optimizer,scheduler,loss
    params=lr_arr=None
    if config.scheduler.type == 'cyclicLR':
        params, lr_arr, _ = discriminative_lr_params(model, slice(1e-5))
    optimizer_q = create_optimizer(config,model_q)
    scheduler_q = create_scheduler(config,
                                  optimizer_q,base_lr=lr_arr,
                                  steps_per_epoch=len(labalbed_dataloader))
    optimizer_k = create_optimizer(config,model_k)
    scheduler_k = create_scheduler(config,
                                  optimizer_k,base_lr=lr_arr,
                                  steps_per_epoch=len(labalbed_dataloader))
    
    #checkpoint
    supervised_loss, Ssl_loss, contra_loss = create_loss(config)

    #

    memory_bank_unlabeled=None
    p_model=PMovingAverage(config.dataset.n_classes,config.model.features_dim)
    
    #supervised train
    #todo
    for epoch, seed in enumerate(range(config.scheduler.epochs)):
        sup_loss=train(config,model,labalbed_dataloader,optimizer_q,supervised_loss,scheduler_q,logger,epoch)
        if epoch%5==0:
        #if epoch=0 we use few shot labeled dataset to get predifined protetype
            if epoch==0:
                emb_sums=predefined_prototype(config,model_q,labalbed_dataloader)
            #if epoch%5==0 we use mixture labeled dataset to get predifined protetype
            else:
                if memory_bank_unlabeled is not None:
                    mixup_loader = get_mixup(memory_bank_unlabeled,num_neighbors)
                    emb_sums=predefined_prototype(config,model_q,mixup_loader)

        pl_loss,memory_bank_unlabeled,mask,_ = unlabeled_train(config,p_model,Ssl_loss,unlabeled_dataloader,model_q,num_neighbors,emb_sums,logger)
        # get_positive_sample()
        # ctr_loss = contrastive_train()
        # total_loss = sup_loss+pl_loss+ctr_loss
        # total_loss.backward()
        



    
    #unlabel dataset
    unlabeled_dataset = ()

    #dataloader
if __name__ == '__main__':
    main()
       
