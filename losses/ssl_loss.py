import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class SslLoss(nn.Module):
    def __init__(self,thred):
        super(SslLoss,self).__init__()
        self.thred=thred
        self.criterion  = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self,input,target):
        #find the max prob > thred
        prob,target = torch.max(target)
        losses=self.criterion(input,target)
        #todo: find a func
        # mask = torch.where(prob,0,1)
        mask = prob > 0.7
        return losses, mask

#test it
if __name__ == '__main__':
    ssl_loss=SslLoss(0.7)