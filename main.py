
from models import build_model
from utils.common_utils import create_config


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
    p=create_config(args.config_exp)

    #build model
    model=build_model(p)
    #print model info
    # print(model.)

    model = model.to(p['device'])  

    #setup cudnn
    # setupCuDNN()

    #label dataset
    
    #unlabel dataset
    unlabeled_dataset = ()

    #dataloader
       
