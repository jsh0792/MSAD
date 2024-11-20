import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.func import *



##########################
#### Genomic FC Model ####
##########################
class SNN(nn.Module):
    def __init__(self, input_dim: int, model_size_omic: str='small', n_classes: int=4):
        super(SNN, self).__init__()
        self.n_classes = n_classes
        self.size_dict_omic = {'small': [256, 256, 256, 256], 'big': [1024, 1024, 1024, 256]}
        
        ### Constructing Genomic SNN
        hidden = self.size_dict_omic[model_size_omic]
        fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
        for i, _ in enumerate(hidden[1:]):
            fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
        self.fc_omic = nn.Sequential(*fc_omic)
        self.classifier = nn.Linear(hidden[-1], n_classes)
        init_max_weights(self)


    def forward(self, **kwargs):
        x = kwargs['all_gene']
        features = self.fc_omic(x)

        logits = self.classifier(features).unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return hazards, S, Y_hat, None, None