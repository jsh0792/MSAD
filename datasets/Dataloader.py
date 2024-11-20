import os
import numpy as np
import torch
from torchvision import datasets, transforms
from BaseClass.BaseDataloader import BaseDataLoader
from datasets.Dataset import SurvDataset
from utils.collate import collate_MIL_survival_sig, collate_MIL_survival_sig_apply_signature_pregene, collate_MIL_survival
from utils.tools import *
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler, SequentialSampler


class SurvDataLoader(BaseDataLoader):
    """
    Survival data loading demo using BaseDataLoader
    """
    def __init__(self, study, csv_dir, data_dir, split_dir, batch_size, fold, data_dir_origin=None, shuffle=True, validation_split=0.0, num_workers=1, training=True, apply_sig=False, use_pre_gene=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        if data_dir_origin is not None:
            data_dir_origin=os.path.join(data_dir_origin, study)
        
        self.dataset = SurvDataset(csv_path = '%s/tcga_%s_all_clean.csv' % (csv_dir, study.lower()),
										   data_dir= os.path.join(data_dir, study),
                                           data_dir_origin=data_dir_origin,
										   shuffle = False, 
										   print_info = True,
										   patient_strat= False,
										   n_bins=4,
										   label_col = 'survival_months',
										   ignore=[],
                                           study = study.lower(),
                                           apply_sig=apply_sig,
                                           use_pre_gene=use_pre_gene)
        train_dataset, val_dataset, test_dataset = self.dataset.return_splits(from_id=False, csv_path='{}/split{}.csv'.format(split_dir, fold))
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.apply_sig = apply_sig
        self.use_pre_gene = use_pre_gene
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    def get_split_loader(self, split_dataset, weighted = True, batch_size=1):
        """
            return either the validation loader or training loader 
        """
        if self.apply_sig==True and self.use_pre_gene==True:
            collate = collate_MIL_survival_sig_apply_signature_pregene
        elif self.apply_sig==True and self.use_pre_gene==False:
            collate = collate_MIL_survival_sig
        else:
            collate = collate_MIL_survival
        
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        kwargs = {'num_workers': 4} if device.type == "cuda" else {}

        if weighted:
            weights = make_weights_for_balanced_classes_split(split_dataset)
            loader = DataLoader(split_dataset, batch_size=batch_size, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate, **kwargs)    
        else:
            loader = DataLoader(split_dataset, batch_size=batch_size, sampler = RandomSampler(split_dataset), collate_fn = collate, **kwargs)

        return loader

    def get_dataloader(self):
        train_dataloader = self.get_split_loader(self.train_dataset)
        val_dataloader = self.get_split_loader(self.val_dataset)
        test_dataloader = self.get_split_loader(self.test_dataset)
        return train_dataloader, val_dataloader, test_dataloader

    def get_gene_num(self):
        return self.train_dataset.get_gene_num()
