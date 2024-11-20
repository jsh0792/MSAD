# 和MACT的区别在于，HiMT有对三个尺度特征进行拼接

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('/data/jsh/dfri_final')
from models.func import *


###########################
### HiMT Implementation ###
###########################
class HiMT(nn.Module):
    def __init__(self, fusion='concat', omic_sizes=[100, 200, 300, 400, 500, 600], n_classes=4, input_dim=1024, 
                 model_size_wsi: str='small', model_size_omic: str='small', dropout=0.25):
        super(HiMT, self).__init__()
        self.fusion = fusion
        self.omic_sizes = omic_sizes
        self.n_classes = n_classes
        self.size_dict_WSI = {"small": [input_dim, 256, 256], "big": [input_dim, 512, 384]}
        # self.size_dict_WSI = {"small": [1000, 256, 256], "big": [1000, 512, 384]}
        self.size_dict_omic = {'small': [256, 256], 'big': [1024, 1024, 1024, 256]}

        ### FC Layer over WSI bag
        size = self.size_dict_WSI[model_size_wsi]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        fc.append(nn.Dropout(0.25))
        self.wsi_net = nn.Sequential(*fc)

        ### Constructing Genomic SNN
        hidden = self.size_dict_omic[model_size_omic]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)

        ### Multihead Attention
        self.coattn = MultiheadAttention(embed_dim=256, num_heads=1)

        ### Path Transformer + Attention Head
        path_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, activation='relu')
        self.path_transformer = nn.TransformerEncoder(path_encoder_layer, num_layers=2)
        self.path_attention_head = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(*[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)])

        ### Omic Transformer + Attention Head
        omic_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, activation='relu')
        self.omic_transformer = nn.TransformerEncoder(omic_encoder_layer, num_layers=2)
        self.omic_attention_head = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        self.omic_rho = nn.Sequential(*[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)])

        ### Fusion Layer
        if self.fusion == 'concat':
            self.mm = nn.Sequential(*[nn.Linear(256*2, size[2]), nn.ReLU(), nn.Linear(size[2], size[2]), nn.ReLU()])
        elif self.fusion == 'bilinear':
            self.mm = BilinearFusion(dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)
        else:
            self.mm = None

        ### Classifier
        self.classifier = nn.Linear(size[2], n_classes)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8)

    def forward(self, **kwargs):
        x_path = kwargs['x_path']
        x_omic = [kwargs['x_omic%d' % i] for i in range(1,7)]

        h_path_bag = self.wsi_net(x_path).unsqueeze(1) ### path embeddings are fed through a FC layer
        # x_path = x_path.cpu()
        h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)] ### each omic signature goes through it's own FC layer

        # add transformer
        h_omic = [self.encoder_layer(sig_feat.unsqueeze(0).unsqueeze(0)) for sig_feat in h_omic]
        h_omic_bag = torch.stack(h_omic).squeeze(1)
        # import pdb; pdb.set_trace()

        # h_omic_bag = torch.stack(h_omic).unsqueeze(1) ### omic embeddings are stacked (to be used in co-attention)

        # Coattn
        h_path_coattn, A_coattn = self.coattn(h_omic_bag, h_path_bag, h_path_bag)

        ### Path
        h_path_trans = self.path_transformer(h_path_coattn)
        A_path, h_path = self.path_attention_head(h_path_trans.squeeze(1))
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1) , h_path)
        h_path = self.path_rho(h_path).squeeze()

        ### Omic
        h_omic_trans = self.omic_transformer(h_omic_bag)
        A_omic, h_omic = self.omic_attention_head(h_omic_trans.squeeze(1))
        A_omic = torch.transpose(A_omic, 1, 0)
        h_omic = torch.mm(F.softmax(A_omic, dim=1) , h_omic)
        h_omic = self.omic_rho(h_omic).squeeze()

        if self.fusion == 'bilinear':
            h = self.mm(h_path.unsqueeze(dim=0), h_omic.unsqueeze(dim=0)).squeeze()
        elif self.fusion == 'concat':
            h = self.mm(torch.cat([h_path, h_omic], axis=0))

        ### Survival Layer
        logits = self.classifier(h).unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        attention_scores = {'coattn': A_coattn, 'path': A_path, 'omic': A_omic}

        return hazards, S, Y_hat, attention_scores


    def captum(self, x_path, x_omic1, x_omic2, x_omic3, x_omic4, x_omic5, x_omic6):
        #x_path = torch.randn((10, 500, 1024))
        #x_omic1, x_omic2, x_omic3, x_omic4, x_omic5, x_omic6 = [torch.randn(10, size) for size in omic_sizes]
        x_omic = [x_omic1, x_omic2, x_omic3, x_omic4, x_omic5, x_omic6]
        h_path_bag = self.wsi_net(x_path)#.unsqueeze(1) ### path embeddings are fed through a FC layer
        h_path_bag = torch.reshape(h_path_bag, (500, 10, 256))
        h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)] ### each omic signature goes through it's own FC layer
        h_omic_bag = torch.stack(h_omic) ### omic embeddings are stacked (to be used in co-attention)

        # Coattn
        h_path_coattn, A_coattn = self.coattn(h_omic_bag, h_path_bag, h_path_bag)

        ### Path
        h_path_trans = self.path_transformer(h_path_coattn)
        h_path_trans = torch.reshape(h_path_trans, (10, 6, 256))
        A_path, h_path = self.path_attention_head(h_path_trans)
        A_path = F.softmax(A_path.squeeze(dim=2), dim=1).unsqueeze(dim=1)
        h_path = torch.bmm(A_path, h_path).squeeze(dim=1)

        ### Omic
        h_omic_trans = self.omic_transformer(h_omic_bag)
        h_omic_trans = torch.reshape(h_omic_trans, (10, 6, 256))
        A_omic, h_omic = self.omic_attention_head(h_omic_trans)
        A_omic = F.softmax(A_omic.squeeze(dim=2), dim=1).unsqueeze(dim=1)
        h_omic = torch.bmm(A_omic, h_omic).squeeze(dim=1)

        if self.fusion == 'bilinear':
            h = self.mm(h_path.unsqueeze(dim=0), h_omic.unsqueeze(dim=0)).squeeze()
        elif self.fusion == 'concat':
            h = self.mm(torch.cat([h_path, h_omic], axis=1))

        logits  = self.classifier(h)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        risk = -torch.sum(S, dim=1)
        return risk

if __name__ == '__main__':
    
    device = torch.device('cuda')
    model = HiMT(omic_sizes=[100, 200, 300, 400, 500, 600]).to(device)
    path_5x = torch.rand(60,1024).to(device)
    path_10x = torch.rand(600,1024).to(device)
    path_20x = torch.rand(9000,1024).to(device)
    x_path = torch.concat([path_10x, path_20x, path_5x], dim=0)
    
    x_omic1 = torch.rand(100).to(device)
    x_omic2 = torch.rand(200).to(device)
    x_omic3 = torch.rand(300).to(device)
    x_omic4 = torch.rand(400).to(device)
    x_omic5 = torch.rand(500).to(device)
    x_omic6 = torch.rand(600).to(device)

    # hazards, S, Y_hat, h_path, gene_5x_loss, gene_10x_loss, gene_20x_loss, gene_vec_snn = mm_model(x_path_5x=path_5x, x_path_10x=path_10x, x_path_20x=path_20x, gene_vec=gene, ablation_geneexp=False)
    hazards, S, Y_hat, attention_scores = model(x_path=x_path, x_omic1 = x_omic1, x_omic2 = x_omic2, x_omic3 = x_omic3, x_omic4 = x_omic4, x_omic5 = x_omic5, x_omic6 = x_omic6)
    print(hazards.shape)