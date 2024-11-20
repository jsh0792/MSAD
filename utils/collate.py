import torch
import numpy as np

def collate_MIL_survival(batch):
    img_5x = torch.cat([item[0] for item in batch], dim = 0)
    img_10x = torch.cat([item[1] for item in batch], dim = 0)
    img_20x = torch.cat([item[2] for item in batch], dim = 0)
    gene = torch.cat([item[3] for item in batch], dim = 0).type(torch.FloatTensor)
    gene_vec = torch.cat([item[4] for item in batch], dim = 0).type(torch.FloatTensor)

    label = torch.LongTensor([item[5] for item in batch])
    event_time = np.array([item[6] for item in batch])
    c = torch.FloatTensor([item[7] for item in batch])
    case_id = np.array([item[8] for item in batch])
    return [img_5x, img_10x, img_20x, gene, gene_vec, label, event_time, c, case_id]


def collate_MIL_survival_sig(batch):
    omic1 =  torch.cat([item[0] for item in batch], dim = 0).type(torch.FloatTensor)
    omic2 =  torch.cat([item[1] for item in batch], dim = 0).type(torch.FloatTensor)
    omic3 =  torch.cat([item[2] for item in batch], dim = 0).type(torch.FloatTensor)
    omic4 =  torch.cat([item[3] for item in batch], dim = 0).type(torch.FloatTensor)
    omic5 =  torch.cat([item[4] for item in batch], dim = 0).type(torch.FloatTensor)
    omic6 =  torch.cat([item[5] for item in batch], dim = 0).type(torch.FloatTensor)
    
    label = torch.LongTensor([item[6] for item in batch])
    event_time = np.array([item[7] for item in batch])
    c = torch.FloatTensor([item[8] for item in batch])
    case_id = np.array([item[9] for item in batch])

    img_5x = torch.cat([item[12] for item in batch], dim = 0)
    img_10x = torch.cat([item[11] for item in batch], dim = 0)
    img_20x = torch.cat([item[10] for item in batch], dim = 0)    
    return [img_5x, img_10x, img_20x, omic1, omic2, omic3, omic4, omic5, omic6, label, event_time, c, case_id]


def collate_MIL_survival_sig_apply_signature_pregene(batch):

    omic1 =  torch.cat([item[0] for item in batch], dim = 0).type(torch.FloatTensor)
    omic2 =  torch.cat([item[1] for item in batch], dim = 0).type(torch.FloatTensor)
    omic3 =  torch.cat([item[2] for item in batch], dim = 0).type(torch.FloatTensor)
    omic4 =  torch.cat([item[3] for item in batch], dim = 0).type(torch.FloatTensor)
    omic5 =  torch.cat([item[4] for item in batch], dim = 0).type(torch.FloatTensor)
    omic6 =  torch.cat([item[5] for item in batch], dim = 0).type(torch.FloatTensor)

    label = torch.LongTensor([item[6] for item in batch])
    event_time = np.array([item[7] for item in batch])
    c = torch.FloatTensor([item[8] for item in batch])
    case_id = np.array([item[9] for item in batch])

    img_5x = torch.cat([item[10] for item in batch], dim = 0)
    img_10x = torch.cat([item[11] for item in batch], dim = 0)
    img_20x = torch.cat([item[12] for item in batch], dim = 0)
    
    gene = torch.cat([item[13] for item in batch], dim = 0).type(torch.FloatTensor)
    gene_vec = torch.cat([item[14] for item in batch], dim = 0).type(torch.FloatTensor)
    return [img_5x, img_10x, img_20x, omic1, omic2, omic3, omic4, omic5, omic6, gene, gene_vec, label, event_time, c, case_id]