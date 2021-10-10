import torch.nn as nn
from .ssl_loss import SslLoss
from .contra_loss import CircleLoss,convert_label_to_similarity










def create_loss(config,reduction):

    supervised_loss = nn.CrossEntropyLoss(reduction='mean')
    ssl_loss = SslLoss(0.7)
    constrastive_loss = CircleLoss(m=0.25, gamma=256)
    

    #supervised_loss, ssl_loss, contractive_loss
    return supervised_loss, ssl_loss, constrastive_loss