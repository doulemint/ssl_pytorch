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
        # print("target: ",target[0]*100)
        target = F.softmax(target,dim=1)#(target)
        # print("target: ",target[0])
        prob,target = torch.max(target,dim=1)
        losses=self.criterion(input,target)
        # mask = torch.where(prob,0,1)
        mask = prob > self.thred
        self.thred += 0.01
        return losses, mask

#test it
if __name__ == '__main__':
    ssl_loss=SslLoss(0.7)