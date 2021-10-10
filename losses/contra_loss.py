import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from typing import Tuple

import torch
from torch import nn, Tensor

import tensorflow.keras.backend as K
def cosine_similarity( x, y):
        x = K.reshape(x, (K.shape(x)[0], -1))
        y = K.reshape(y, (K.shape(y)[0], -1))
        abs_x = K.sqrt(K.sum(K.square(x), axis=1, keepdims=True))
        abs_y = K.sqrt(K.sum(K.square(y), axis=1, keepdims=True))
        up = K.dot(x, K.transpose(y))
        down = K.dot(abs_x, K.transpose(abs_y))
        return up / down

def convert_mlphead_to_similarity(normed_feature: Tensor,label: Tensor, emd_sum: Tensor,model: nn.modules):
    #calculate similarity with predefined protetype rather than within mini-batch size
    #todo: replace it to cos sim
    #for delta_p
    similarity_matrix = normed_feature @ emd_sum

    similarity_matrix2 = normed_feature @ model.queue.tranpose(1,0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)





def convert_label_to_similarity(normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    # print("similarity_matrix: ",similarity_matrix)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)
    # print("label_matrix: ",label_matrix)

    positive_matrix = label_matrix.triu(diagonal=1)
    # print("positive_matrix: ",positive_matrix)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)
    # print("negative_matrix: ",negative_matrix)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]


class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)
        # print("ap:",ap," an:",an)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma
        # print("logit_p:",logit_p," logit_n:",logit_n)

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss


if __name__ == "__main__":
    feat = nn.functional.normalize(torch.rand(5, 10, requires_grad=True))
    lbl = torch.randint(high=10, size=(5,))
    print("feat: ",feat)
    print("lbl: ",lbl)

    inp_sp, inp_sn = convert_label_to_similarity(feat, lbl)
    print("similarity positive: ",inp_sp)
    print("similarity negative: ",inp_sn)

    criterion = CircleLoss(m=0.25, gamma=256)
    circle_loss = criterion(inp_sp, inp_sn)

    # print(circle_loss)
