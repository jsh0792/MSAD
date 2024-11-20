from __future__ import print_function, division
import math
import os
import pdb
import pickle
import re

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset
from pathlib import Path


class SurvDataset(Dataset):
    def __init__(self,
        csv_path = 'dataset_csv/ccrcc_clean.csv', data_dir=None, data_dir_origin=None, apply_sig = False, use_pre_gene=True, 
        shuffle = False, seed = 7, print_info = True, n_bins = 4, ignore=[], study = 'brca',
        patient_strat=False, label_col = None, filter_dict = {}, eps=1e-6):
        r"""
        Generic_WSI_Survival_Dataset 

        Args:
            csv_file (string): Path to the csv file with annotations.
            shuffle (boolean): Whether to shuffle
            seed (int): random seed for shuffling the data
            print_info (boolean): Whether to print a summary of the dataset
            label_dict (dict): Dictionary with key, value pairs for converting str labels to int
            ignore (list): List containing class labels to ignore
        """
        super(SurvDataset, self).__init__()
        self.custom_test_ids = None
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
        self.data_dir = data_dir
        self.data_dir_origin = data_dir_origin
        self.study = study
        self.use_pre_gene = use_pre_gene
        
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)

        slide_data = pd.read_csv(csv_path, low_memory=False)
        ### 筛选有效数据
        tmp_set = set([item[:-4] for item in slide_data['slide_id'].values])
        # if data_dir.split('/')[-1] == 'LGG' or data_dir.split('/')[-1] == 'GBM':
        #     data_dir = os.path.join(str(Path(data_dir).parent), 'GBMLGG')
        
        if self.data_dir_origin is not None:
            slide_data_existed = set([os.path.splitext(os.path.basename(file))[0] for file in os.listdir(os.path.join(data_dir_origin, '20x/Ctranspath/pt_files'))]) 
        else:
            slide_data_existed = set([os.path.splitext(os.path.basename(file))[0] for file in os.listdir(os.path.join(data_dir, '5x_fea/cluster_fea'))])
        
        
        intersec_set = tmp_set & slide_data_existed
        slide_data_new = [item + '.svs' for item in intersec_set]
        slide_data = slide_data.loc[slide_data['slide_id'].isin(slide_data_new)].reset_index()

        ### 筛选有效数据
        tmp_set = set([item[:12] + '-01' for item in slide_data['slide_id'].values])
        gene_pt = torch.load("/data/jsh/dfri_final/data/processed_gene/normalized/{}.pt".format(study))
        gene_data_existed = set(gene_pt.keys())
        intersec_set = tmp_set & gene_data_existed
        slide_data_new = [item + '.svs' for item in intersec_set]
        slide_data = slide_data[slide_data['slide_id'].apply(lambda x: any(x.startswith(prefix[:-3]) for prefix in intersec_set))].reset_index()

        #slide_data = slide_data.drop(['Unnamed: 0'], axis=1)
        if 'case_id' not in slide_data:
            slide_data.index = slide_data.index.str[:12]
            slide_data['case_id'] = slide_data.index
            slide_data = slide_data.reset_index(drop=True)
        import pdb
        #pdb.set_trace()

        if not label_col:
            label_col = 'survival_months'
        else:
            assert label_col in slide_data.columns
        self.label_col = label_col

        # 因为BRCA数据集结果差，去掉这个判断条件
        # if "IDC" in slide_data['oncotree_code']: # must be BRCA (and if so, use only IDCs)
        #     slide_data = slide_data[slide_data['oncotree_code'] == 'IDC']

        patients_df = slide_data.drop_duplicates(['case_id']).copy()
        uncensored_df = patients_df[patients_df['censorship'] < 1] # 1代表删失

        disc_labels, q_bins = pd.qcut(uncensored_df[label_col], q=n_bins, retbins=True, labels=False)
        q_bins[-1] = slide_data[label_col].max() + eps
        q_bins[0] = slide_data[label_col].min() - eps
        
        disc_labels, q_bins = pd.cut(patients_df[label_col], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        patients_df.insert(2, 'label', disc_labels.values.astype(int))

        patient_dict = {}
        slide_data = slide_data.set_index('case_id')
        slide_cnt = 0
        for patient in patients_df['case_id']:
            slide_ids = slide_data.loc[patient, 'slide_id']
            if isinstance(slide_ids, str):
                slide_ids = np.array(slide_ids).reshape(-1)
            else:
                slide_ids = slide_ids.values
            slide_cnt += len(slide_ids)
            patient_dict.update({patient:slide_ids})

        self.patient_dict = patient_dict
    
        slide_data = patients_df
        slide_data.reset_index(drop=True, inplace=True)
        slide_data = slide_data.assign(slide_id=slide_data['case_id'])

        label_dict = {}
        key_count = 0
        for i in range(len(q_bins)-1):
            for c in [0, 1]:
                print('{} : {}'.format((i, c), key_count))
                label_dict.update({(i, c):key_count})
                key_count+=1

        self.label_dict = label_dict
        for i in slide_data.index:
            key = slide_data.loc[i, 'label']
            slide_data.at[i, 'disc_label'] = key
            censorship = slide_data.loc[i, 'censorship']
            key = (key, int(censorship))
            slide_data.at[i, 'label'] = label_dict[key]

        self.bins = q_bins
        self.num_classes=len(self.label_dict)
        patients_df = slide_data.drop_duplicates(['case_id'])
        self.patient_data = {'case_id':patients_df['case_id'].values, 'label':patients_df['label'].values}

        new_cols = list(slide_data.columns[-1:]) + list(slide_data.columns[:-1])
        slide_data = slide_data[new_cols]
        self.slide_data = slide_data
        self.metadata = slide_data.columns[:12]
        self.cls_ids_prep()
        self.align_patient_gene()

        # if print_info:
        #     self.summarize()

        ### Signatures
        self.apply_sig = apply_sig
        if self.apply_sig:
            self.signatures = pd.read_csv('signatures.csv')
        else:
            self.signatures = None

        if print_info:
            self.summarize()
        print(slide_data)
        print('censor==0')
        print(len(slide_data[slide_data['censorship']==0]))
        print('censor==1')
        print(len(slide_data[slide_data['censorship']==1]))

    def cls_ids_prep(self):
        self.patient_cls_ids = [[] for i in range(self.num_classes)]        
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]


    def patient_data_prep(self):
        patients = np.unique(np.array(self.slide_data['case_id'])) # get unique patients
        patient_labels = []
        
        for p in patients:
            locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
            assert len(locations) > 0
            label = self.slide_data['label'][locations[0]] # get patient label
            patient_labels.append(label)
        
        self.patient_data = {'case_id':patients, 'label':np.array(patient_labels)}


    @staticmethod
    def df_prep(data, n_bins, ignore, label_col):
        mask = data[label_col].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        disc_labels, bins = pd.cut(data[label_col], bins=n_bins)
        return data, bins

    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data['case_id'])
        else:
            return len(self.slide_data)

    def summarize(self):
        print("label column: {}".format(self.label_col))
        print("label dictionary: {}".format(self.label_dict))
        print("number of classes: {}".format(self.num_classes))
        print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))
        for i in range(self.num_classes):
            print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
            print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))


    def get_split_from_df(self, all_splits: dict, split_key: str='train', scaler=None):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(split.tolist())
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = Generic_Split(df_slice, metadata=self.metadata, study=self.study, signatures=self.signatures, data_dir=self.data_dir, 
                                  data_dir_origin=self.data_dir_origin, label_col=self.label_col, patient_dict=self.patient_dict,
                                  num_classes=self.num_classes, cancer_gene_vector=self.cancer_gene_vector, patient_gene=self.patient_gene,
                                  pretrained_embedding=self.pretrained_embedding, use_pre_gene=self.use_pre_gene)
        else:
            split = None
        
        return split


    def return_splits(self, from_id: bool=True, csv_path: str=None):
        if from_id:
            raise NotImplementedError
        else:
            assert csv_path 
            all_splits = pd.read_csv(csv_path)
            train_split = self.get_split_from_df(all_splits=all_splits, split_key='train')
            val_split = self.get_split_from_df(all_splits=all_splits, split_key='val')
            test_split = self.get_split_from_df(all_splits=all_splits, split_key='test')

            ### --> Normalizing Data
            print("****** Normalizing Data ******")
            scalers = train_split.get_scaler()
            train_split.apply_scaler(scalers=scalers)
            val_split.apply_scaler(scalers=scalers)
            test_split.apply_scaler(scalers=scalers)
            ### <--
        return train_split, val_split, test_split


    def get_list(self, ids):
        return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
        return self.slide_data['label'][ids]

    def align_patient_gene(self):
        # 提取出特定癌症的pretrained_embedding
        self.pretrained_embedding = torch.load('gene_interaction.pt')
        self.patient_gene = torch.load('data/processed_gene/normalized/{}.pt'.format(self.data_dir.split('/')[-1].lower()))
        self.cancer_gene_vector = []
        self.gene_idx_list = []
        for gene_name in self.patient_gene['gene']:
            self.cancer_gene_vector.append(np.array(self.pretrained_embedding[gene_name]))
        self.cancer_gene_vector = torch.from_numpy(np.array(self.cancer_gene_vector))

    def get_gene_num(self):
        return len(self.patient_gene['gene'])

    def __getitem__(self, idx):
        case_id = self.slide_data['case_id'][idx]
        label = self.slide_data['disc_label'][idx]
        event_time = self.slide_data[self.label_col][idx]
        c = self.slide_data['censorship'][idx]
        slide_ids = self.patient_dict[case_id]
        case_gene = torch.from_numpy(self.patient_gene[case_id + '-01'])

        if type(self.data_dir) == dict:
            source = self.slide_data['oncotree_code'][idx]
            data_dir = self.data_dir[source]
        else:
            data_dir = self.data_dir
        # if data_dir.split('/')[-1] == 'LGG' or data_dir.split('/')[-1] == 'GBM':
        #     data_dir = os.path.join(str(Path(data_dir).parent), 'GBMLGG')
        
        ### 控制Dataset输出的有两个, signature控制是否有omic1_6, use_pre_gene控制是否加入预训练的cancer_gene_vec
        if self.signatures is not None and self.use_pre_gene==False:
            path_features_20x = []

            omic1 = torch.tensor(self.genomic_features[self.omic_names[0]].iloc[idx])   # self.omic_names是根据signature.csv文件的划分结果得到的
            omic2 = torch.tensor(self.genomic_features[self.omic_names[1]].iloc[idx])
            omic3 = torch.tensor(self.genomic_features[self.omic_names[2]].iloc[idx])
            omic4 = torch.tensor(self.genomic_features[self.omic_names[3]].iloc[idx])
            omic5 = torch.tensor(self.genomic_features[self.omic_names[4]].iloc[idx])
            omic6 = torch.tensor(self.genomic_features[self.omic_names[5]].iloc[idx])
            
            path_features_5x = []
            path_features_10x = []
            path_features_20x = []
            for slide_id in slide_ids:
                if self.data_dir_origin is not None:    # 不使用聚类的特征
                    wsi_path_5x  = os.path.join(self.data_dir_origin, '5x', 'Ctranspath', 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                    wsi_path_10x = os.path.join(self.data_dir_origin, '10x', 'Ctranspath', 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                    wsi_path_20x = os.path.join(self.data_dir_origin, '20x', 'Ctranspath', 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                else:                    
                    wsi_path_5x  = os.path.join(data_dir, '5x_fea', 'cluster_fea', '{}.pt'.format(slide_id.rstrip('.svs')))
                    wsi_path_10x = os.path.join(data_dir, '5x_fea', 'cluster_fea', '{}.pt'.format(slide_id.rstrip('.svs')))
                    wsi_path_20x = os.path.join(data_dir, '5x_fea', 'cluster_fea', '{}.pt'.format(slide_id.rstrip('.svs')))
                    
                wsi_bag_5x = torch.load(wsi_path_5x).clone().detach().type(torch.float32)
                wsi_bag_10x = torch.load(wsi_path_10x).clone().detach().type(torch.float32)
                wsi_bag_20x = torch.load(wsi_path_20x).clone().detach().type(torch.float32)
                
                path_features_5x.append(wsi_bag_5x)
                path_features_10x.append(wsi_bag_10x)
                path_features_20x.append(wsi_bag_20x)
                
            path_features_5x = torch.cat(path_features_5x, dim=0)
            path_features_10x = torch.cat(path_features_10x, dim=0)
            path_features_20x = torch.cat(path_features_20x, dim=0)
                
            return (omic1, omic2, omic3, omic4, omic5, omic6, label, event_time, c, case_id, path_features_20x, path_features_10x, path_features_5x)
        
        if not self.use_h5:
            if self.data_dir:
                path_features_5x = []
                path_features_10x = []
                path_features_20x = []
                for slide_id in slide_ids:
                    if self.data_dir_origin is not None:    # DFRI_v3
                        wsi_path_5x  = os.path.join(self.data_dir_origin, '5x', 'Ctranspath', 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_path_10x = os.path.join(self.data_dir_origin, '10x', 'Ctranspath', 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_path_20x = os.path.join(self.data_dir_origin, '20x', 'Ctranspath', 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                    else:                    
                        wsi_path_5x  = os.path.join(data_dir, '5x_fea', 'cluster_fea', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_path_10x = os.path.join(data_dir, '5x_fea', 'cluster_fea', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_path_20x = os.path.join(data_dir, '5x_fea', 'cluster_fea', '{}.pt'.format(slide_id.rstrip('.svs')))
                    
                    ### choice 1 (when use the cluster feature)
                    # wsi_bag_5x = torch.tensor(torch.load(wsi_path_5x), dtype=torch.float32)  
                    # wsi_bag_10x = torch.tensor(torch.load(wsi_path_10x), dtype=torch.float32)  
                    # wsi_bag_20x = torch.tensor(torch.load(wsi_path_20x), dtype=torch.float32)
                    
                    ### choice 2
                    wsi_bag_5x = torch.load(wsi_path_5x).clone().detach().type(torch.float32)
                    wsi_bag_10x = torch.load(wsi_path_10x).clone().detach().type(torch.float32)
                    wsi_bag_20x = torch.load(wsi_path_20x).clone().detach().type(torch.float32)
                    
                    # 可以选择跳过这个样本，或者返回一个预定义的错误值
                    # /data/jsh/TCGA/LUAD/20x/Ctranspath/pt_files/TCGA-97-A4M5-01Z-00-DX1.283AFBA6-A349-425F-A18E-5CA186084C23.pt
                    # /TCGA-95-A4VK-01Z-00-DX1.D09778E0-285E-4593-84C8-B6009DDF4E41.pt
                    path_features_5x.append(wsi_bag_5x)
                    path_features_10x.append(wsi_bag_10x)
                    path_features_20x.append(wsi_bag_20x)

                path_features_5x = torch.cat(path_features_5x, dim=0)
                path_features_10x = torch.cat(path_features_10x, dim=0)
                path_features_20x = torch.cat(path_features_20x, dim=0)
                
                if self.signatures is not None:
                    omic1 = torch.tensor(self.genomic_features[self.omic_names[0]].iloc[idx])   # self.omic_names是根据signature.csv文件的划分结果得到的
                    omic2 = torch.tensor(self.genomic_features[self.omic_names[1]].iloc[idx])
                    omic3 = torch.tensor(self.genomic_features[self.omic_names[2]].iloc[idx])
                    omic4 = torch.tensor(self.genomic_features[self.omic_names[3]].iloc[idx])
                    omic5 = torch.tensor(self.genomic_features[self.omic_names[4]].iloc[idx])
                    omic6 = torch.tensor(self.genomic_features[self.omic_names[5]].iloc[idx])
                    # print('-'*50)
                    return (omic1, omic2, omic3, omic4, omic5, omic6, label, event_time, c, case_id, path_features_5x, path_features_10x, path_features_20x, case_gene, self.cancer_gene_vector)
                else:
                    return (path_features_5x, path_features_10x, path_features_20x, case_gene, self.cancer_gene_vector, label, event_time, c, case_id)
            else:
                return slide_ids, label, event_time, c
      


class Generic_Split(SurvDataset):
    def __init__(self, slide_data, metadata, study, signatures=None, data_dir=None, data_dir_origin=None, label_col=None, patient_dict=None, num_classes=2, cancer_gene_vector=None, 
                 patient_gene=None, use_pre_gene=True, pretrained_embedding=None):
        self.use_h5 = False
        self.slide_data = slide_data
        self.metadata = metadata
        self.study = study
        self.data_dir = data_dir
        self.data_dir_origin = data_dir_origin
        self.num_classes = num_classes
        self.label_col = label_col
        self.patient_dict = patient_dict
        self.cancer_gene_vector = cancer_gene_vector
        self.patient_gene = patient_gene
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

        ### --> Initializing genomic features in Generic Split
        self.genomic_features = self.slide_data.drop(self.metadata, axis=1)
        self.pretrained_embedding =  pretrained_embedding
        self.signatures = signatures
        self.use_pre_gene = use_pre_gene
        def series_intersection(s1, s2):
            return pd.Series(list(set(s1) & set(s2)))

        if self.signatures is not None:
            self.omic_names = []
            for col in self.signatures.columns:
                omic = self.signatures[col].dropna().unique()
                if self.study.lower() == 'stad' or  self.study.lower() == 'ucec' or  self.study.lower() == 'kirc':
                    omic = np.concatenate([omic+mode for mode in ['_mut', '_cnv', '_rnaseq', '']])  # temp add, because the download gene don't have the kind
                else:
                    omic = np.concatenate([omic+mode for mode in ['_mut', '_cnv', '_rnaseq']])  # temp add, because the download gene don't have the kind
                omic = sorted(series_intersection(omic, self.genomic_features.columns))
                self.omic_names.append(omic)
            
            ## new add, pre-trained gene
            self.pre_omic_names = []
            for col in self.signatures.columns:
                omic = self.signatures[col].dropna().unique()
                omic = np.concatenate([omic+mode for mode in ['_mut', '_cnv', '_rnaseq', '']])  # temp add, because the download gene don't have the kind
                omic = sorted(series_intersection(omic, self.pretrained_embedding.keys()))
                self.pre_omic_names.append(omic)
            
            self.omic_sizes = [len(omic) for omic in self.omic_names]
            print('-----------------------self.omic_sizes-------------------------------' )
            print(self.omic_sizes)
        # print("Shape", self.genomic_features.shape)
        ### 获得所有基因名称


    def __len__(self):
        return len(self.slide_data)

    ### --> Getting StandardScaler of self.genomic_features
    def get_scaler(self):
        scaler_omic = StandardScaler().fit(self.genomic_features)
        return (scaler_omic,)
    ### <--

    ### --> Applying StandardScaler to self.genomic_features
    def apply_scaler(self, scalers: tuple=None):
        transformed = pd.DataFrame(scalers[0].transform(self.genomic_features))
        transformed.columns = self.genomic_features.columns
        self.genomic_features = transformed
    ### <--