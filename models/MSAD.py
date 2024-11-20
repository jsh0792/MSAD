from collections import OrderedDict
from os.path import join
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('MASD')
from models.func import *
from tqdm import tqdm
# from thop import profile

class MSAD(nn.Module):
    def __init__(self, fusion='concat', n_classes=4, all_gene_dim=10000,
                 model_size_wsi: str='small', model_size_omic: str='small', dropout=0.25, use_cossim=False, input_dim=768, 
                 train_visual=False, study='LUAD', ablation_k=6):
        super(MSAD, self).__init__()
        self.fusion = fusion
        self.n_classes = n_classes
        self.use_cossim = use_cossim
        self.size_dict_WSI = {"small": [input_dim, 256, 256], "big": [input_dim, 512, 384]}
        self.size_dict_omic = {'small': [256, 256], 'big': [1024, 1024, 1024, 256]}
        self.train_visual = train_visual
        self.study = study
        self.ablation_k = ablation_k
        
        ### FC Layer over WSI bag
        size = self.size_dict_WSI[model_size_wsi]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(0.25)]
        self.wsi_net_5x = nn.Sequential(*fc)
        self.wsi_net_10x = nn.Sequential(*fc)
        self.wsi_net_20x = nn.Sequential(*fc)

        #gene_fc = [nn.Linear(200, size[1])]
        fine_tune_layer = [nn.Linear(size[1], size[1]), nn.LeakyReLU()]
        self.gene_proj_net = nn.Linear(200, size[1])
        self.fine_tune_net_1 = nn.Sequential(*fine_tune_layer)
        self.fine_tune_net_2 = nn.Sequential(*fine_tune_layer)
        self.fine_tune_net_3 = nn.Sequential(*fine_tune_layer)

        ### Multihead Attention
        self.coattn_5x = MultiheadAttention(embed_dim=256, num_heads=1)
        self.coattn_10x = MultiheadAttention(embed_dim=256, num_heads=1)
        self.coattn_20x = MultiheadAttention(embed_dim=256, num_heads=1)

        self.mask_5x = nn.Parameter((torch.ones(all_gene_dim, requires_grad=True)))
        #self.mask_5x = nn.init.normal_(self.mask_5x)
        self.mask_10x = nn.Parameter((torch.ones(all_gene_dim, requires_grad=True)))
        #self.mask_10x = nn.init.normal_(self.mask_10x)
        self.mask_20x = nn.Parameter((torch.ones(all_gene_dim, requires_grad=True)))
        #self.mask_20x = nn.init.normal_(self.mask_20x)

        #self.ablation_geneexp_proj = nn.Linear(1, 200)
        # self.ablation_geneexp_proj = nn.Linear(all_gene_dim, 256)

        self.gene_5x_proj = nn.Linear(256, 1)
        self.gene_10x_proj = nn.Linear(256, 1)
        self.gene_20x_proj = nn.Linear(256, 1)

        self.gene_vec_bias = nn.Parameter((torch.zeros(all_gene_dim, 200, requires_grad=True)))
        # nn.init.constant_(self.gene_vec_bias, 0.1)

        self.scale_attention_head = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        self.scale_rho = nn.Sequential(*[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)])

        self.gene_attention_head = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        self.gene_rho = nn.Sequential(*[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)])

        ### [new add] filter-gate-attention
        attention_net = Attn_Net_Gated(L = size[1], D = size[1], dropout = dropout, n_classes = 1)
        self.attention_net_5x = attention_net
        self.attention_net_10x = attention_net
        self.attention_net_20x = attention_net

        self.h0 = nn.Parameter(torch.randn(3, all_gene_dim, 256))
        self.c0 = nn.Parameter(torch.randn(3, all_gene_dim, 256))

        
        ### Classifier
        self.classifier = nn.Linear(size[2], n_classes)

        # [new-version] top-to-down attention
        self.W_wf_5x = nn.Parameter(torch.rand(256, 1))
        self.b_wf_5x = nn.Parameter(torch.rand(1))
        self.W_wf_10x = nn.Parameter(torch.rand(256, 1))
        self.b_wf_10x = nn.Parameter(torch.rand(1))   

    def forward(self, **kwargs):
        x_path_5x = kwargs['x_path_5x']
        x_path_10x = kwargs['x_path_10x']
        x_path_20x = kwargs['x_path_20x']
        case_id = kwargs['case_id']
        
        gene_vec = kwargs['gene_vec']
        ablation_geneexp = kwargs['ablation_geneexp']

        if ablation_geneexp:
             gene_vec = gene_vec.unsqueeze(0)
             gene_vec_snn = None
        else:
            gene_vec = gene_vec + self.gene_vec_bias
            gene_vec_snn = gene_vec
            gene_vec = self.gene_proj_net(gene_vec).unsqueeze(1)
            gene_vec = self.fine_tune_net_1(gene_vec)

        x_path_5x = self.wsi_net_5x(x_path_5x).unsqueeze(1) # (6,256)
        x_path_10x = self.wsi_net_10x(x_path_10x).unsqueeze(1)
        x_path_20x = self.wsi_net_20x(x_path_20x).unsqueeze(1)

        if x_path_5x.shape[0] < self.ablation_k:
            x_path_5x = x_path_10x
        A, h = self.attention_net_5x(x_path_5x)  # NxK
        topk_indexes_5x = torch.topk(A.squeeze(), self.ablation_k, sorted=True)[1]        
        A, h = self.attention_net_10x(x_path_10x)  # NxK
        topk_indexes_10x = torch.topk(A.squeeze(), self.ablation_k, sorted=True)[1]
        A, h = self.attention_net_20x(x_path_20x)  # NxK
        topk_indexes_20x = torch.topk(A.squeeze(), self.ablation_k, sorted=True)[1]   

        x_path_5x = x_path_5x[topk_indexes_5x]
        x_path_10x = x_path_10x[topk_indexes_10x]
        x_path_20x = x_path_20x[topk_indexes_20x]

        # Coattn
        coattn_5x, Attn_map_5x = self.coattn_5x(gene_vec, x_path_5x, x_path_5x) # (17313, 256) (17313, 6)
        coattn_10x, Attn_map_10x = self.coattn_10x(gene_vec, x_path_10x, x_path_10x)
        coattn_20x, Attn_map_20x = self.coattn_20x(gene_vec, x_path_20x, x_path_20x)

        if self.use_cossim:
            Attn_map_5x = Attn_map_5x.squeeze()
            Attn_map_10x = Attn_map_10x.squeeze()
            Attn_map_20x = Attn_map_20x.squeeze()

            gene_mask_5x = torch.mean(Attn_map_5x, dim=1)
            gene_mask_10x = torch.mean(Attn_map_10x, dim=1)
            gene_mask_20x = torch.mean(Attn_map_20x, dim=1)

            gene_5x = gene_vec[torch.topk(gene_mask_5x, 3000).indices]
            gene_10x = gene_vec[torch.topk(gene_mask_10x, 3000).indices]
            gene_20x = gene_vec[torch.topk(gene_mask_20x, 3000).indices]


            gene_5x_instance = self.gene_5x_proj(gene_5x).squeeze(2) #(N, 1)
            gene_10x_instance = self.gene_10x_proj(gene_10x).squeeze(2)
            gene_20x_instance = self.gene_20x_proj(gene_20x).squeeze(2)

            _, m_indices = torch.sort(gene_5x_instance, 0, descending=True)
            m_feats_5x = torch.index_select(gene_5x, dim=0, index=m_indices[0, :]).squeeze(0)   # 在3000个基因中，选出一个关键gene，进一步进行相似性比较
            if self.train_visual:
                os.makedirs(os.path.join('MSAD/visualization/{}/fold0/'.format(self.study), str(case_id)), exist_ok=True)
                torch.save(torch.topk(gene_mask_5x, 3000).indices.squeeze(), os.path.join('visualization/{}/fold0/'.format(self.study), str(case_id), '5x_gene_index.pt') )

            _, m_indices = torch.sort(gene_10x_instance, 0, descending=True)
            m_feats_10x = torch.index_select(gene_10x, dim=0, index=m_indices[0, :]).squeeze(0)
            if self.train_visual:
                torch.save(torch.topk(gene_mask_10x, 3000).indices.squeeze(), os.path.join('MSAD/visualization/{}/fold0/'.format(self.study), str(case_id), '10x_gene_index.pt') )

            _, m_indices = torch.sort(gene_20x_instance, 0, descending=True)
            m_feats_20x = torch.index_select(gene_20x, dim=0, index=m_indices[0, :]).squeeze(0)
            if self.train_visual:
                torch.save(torch.topk(gene_mask_20x, 3000).indices.squeeze(), os.path.join('/data/jsh/dfri_final/visualization/{}/fold0/'.format(self.study),str(case_id), '20x_gene_index.pt') )
                torch.save(gene_vec.squeeze(), os.path.join('visualization/{}/fold0/'.format(self.study), str(case_id), 'gene_vec_after.pt') )

            gene_5x_similarity = []
            gene_10x_similarity = []
            gene_20x_similarity = []
            for gene in gene_5x:
                gene_5x_similarity.append(F.cosine_similarity(m_feats_5x, gene))
                
            for gene in gene_10x:
                gene_10x_similarity.append(F.cosine_similarity(m_feats_10x, gene))
                
            for gene in gene_20x:
                gene_20x_similarity.append(F.cosine_similarity(m_feats_20x, gene))
                
            gene_5x_similarity = torch.cat(gene_5x_similarity)
            gene_10x_similarity = torch.cat(gene_10x_similarity)
            gene_20x_similarity = torch.cat(gene_20x_similarity)

            gene_5x_wMatrix = gene_5x_similarity[torch.topk(gene_5x_similarity, 2000).indices]
            gene_10x_wMatrix = gene_10x_similarity[torch.topk(gene_10x_similarity, 2000).indices]
            gene_20x_wMatrix = gene_20x_similarity[torch.topk(gene_20x_similarity, 2000).indices]

            gene_5x_wMatrix = 1 - gene_5x_wMatrix
            gene_10x_wMatrix = 1 - gene_10x_wMatrix
            gene_20x_wMatrix = 1 - gene_20x_wMatrix

            gene_5x_loss = torch.sum(gene_5x_wMatrix)
            gene_10x_loss = torch.sum(gene_10x_wMatrix)
            gene_20x_loss = torch.sum(gene_20x_wMatrix)


        if ablation_geneexp:
            multiscale_coattn_seq = torch.stack((coattn_5x, coattn_10x, coattn_20x), dim=0).squeeze(1)
        else:
            multiscale_coattn_seq = torch.stack((coattn_5x, coattn_10x, coattn_20x), dim=0).squeeze()
        
        ### Path
        #去掉gene维度
        A_path, h_path = self.gene_attention_head(multiscale_coattn_seq)
        A_path = torch.transpose(A_path, 1, 2)
        h_path = torch.bmm(F.softmax(A_path, dim=2) , h_path)
        h_path = self.gene_rho(h_path).squeeze()

        h_path_5x = h_path[0]   # (1, 256)
        h_path_10x = h_path[1]
        h_path_20x = h_path[2]     
        
        c_5x = torch.sigmoid(torch.matmul(h_path_10x, self.W_wf_5x) + self.b_wf_5x)
        h_path_10x = h_path_10x + c_5x*h_path_5x
        c_10x = torch.sigmoid(torch.matmul(h_path_20x, self.W_wf_10x) +  self.b_wf_10x)
        h_path_20x = h_path_20x + c_10x*h_path_10x

        h_path = h_path_20x

        ### Survival Layer
        logits = self.classifier(h_path).unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        
        if self.use_cossim:
            return hazards, S, Y_hat, h_path, gene_5x_loss, gene_10x_loss, gene_20x_loss, gene_vec_snn
        else:
            return hazards, S, Y_hat, h_path, None, None, None, gene_vec_snn

class SNN(nn.Module):
    def __init__(self, all_gene_dim, model_size_omic: str='small', n_classes: int=4):
        super(SNN, self).__init__()
        self.n_classes = n_classes
        self.size_dict_omic = {'small': [256, 256, 256, 1], 'big': [1024, 1024, 1024, 256]}
        
        ### Constructing Genomic SNN
        hidden = self.size_dict_omic[model_size_omic]
        fc_omic = [SNN_Block(dim1=200, dim2=hidden[0])]
        for i, _ in enumerate(hidden[1:]):
            fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
        self.fc_omic = nn.Sequential(*fc_omic)
        self.aggr = nn.Sequential(*[SNN_Block(dim1=all_gene_dim, dim2=1024, dropout=0.25),
                                    SNN_Block(dim1=1024, dim2=256, dropout=0.25)])
        self.classifier = nn.Linear(256, n_classes)
        init_max_weights(self)


    def forward(self, **kwargs):
        x = kwargs['gene_vec']
        ablation_geneexp = kwargs['ablation_geneexp']
        if not ablation_geneexp:
            features = self.fc_omic(x).squeeze()
            features = self.aggr(features)
        else:
            features = self.aggr(x)

        logits = self.classifier(features).unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return hazards, S, Y_hat, features

class AggrLayer(nn.Module):
    def __init__(self, embed_dim=256, n_classes: int=4):
        super(AggrLayer, self).__init__()
        self.n_classes = n_classes
        fc = [  nn.Linear(embed_dim*2, embed_dim), nn.ReLU(), nn.Dropout(0.25),
                nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Dropout(0.25),
                nn.Linear(embed_dim, embed_dim)]
        self.fusion_layer = nn.Sequential(*fc)
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, **kwargs):
        gene_embed = kwargs['gene_embedding']
        path_embed = kwargs['mm_embedding']
        features = self.fusion_layer(torch.cat((gene_embed, path_embed)))

        logits = self.classifier(features).unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return hazards, S, Y_hat

if __name__ == '__main__':
    device = torch.device('cuda')
    mm_model = MSAD(input_dim=1024, all_gene_dim=17313, use_cossim=True).to(device)
    gene_model = SNN(all_gene_dim=17313).to(device)
    aggr_model = AggrLayer().to(device)
    gene = torch.rand(17313,200).to(device)
    path_5x = torch.rand(60,1024).to(device)
    path_10x = torch.rand(600,1024).to(device)
    path_20x = torch.rand(9000,1024).to(device)
    mm_embedding = torch.rand(256).to(device)
    gene_embedding = torch.rand(256).to(device)
    
    hazards, S, Y_hat, h_path, gene_5x_loss, gene_10x_loss, gene_20x_loss, gene_vec_snn = mm_model(x_path_5x=path_5x, x_path_10x=path_10x, x_path_20x=path_20x, gene_vec=gene, ablation_geneexp=False)
    # hazards_gene, S_gene, Y_hat, gene_embedding  = gene_model(gene_vec=gene)
    # hazards_aggr, S_aggr, Y_hat  = aggr_model(mm_embedding=mm_embedding, gene_embedding=gene_embedding)

