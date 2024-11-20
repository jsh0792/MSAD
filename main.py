import torch
import collections
import argparse
import importlib
import numpy as np
import torch.optim as optim
from utils.parse import ConfigParser
from Trainer.Trainer import SurvTrainer
from Trainer.Trainer_v4 import SurvTrainer_v4
from Trainer.Trainer_WSI import SurvTrainer_WSI
from Trainer.Trainer_MCAT import SurvTrainer_MCAT
from Trainer.Trainer_HiMT import SurvTrainer_HiMT
from Trainer.Trainer_CSMIL import SurvTrainer_CSMIL
from Trainer.Trainer_DSMIL import SurvTrainer_DSMIL
from datasets.Dataset import SurvDataset
import datasets.Dataloader as module_data
import utils.loss as module_loss
import utils.metric as module_metric
from utils.tools import prepare_device
from models.DFRI import SNN, AggrLayer, SNN_IG

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    folds = np.arange(0, config['trainer']['fold'])

	### Start 5-Fold CV Evaluation.
    for i in folds:
        logger = config.get_logger('train')

        # 初始化dataset 
        data_loader = config.init_obj('data_loader', module_data, fold=i)
        train_dataloader, val_dataloader, test_dataloader = data_loader.get_dataloader()
        gene_dim = data_loader.get_gene_num()

        # 初始化模型结构
        module_arch = importlib.import_module('models.{}'.format(config['name']))
              
        if config['name'][0:4] == 'MSAD':
            model1 = config.init_obj('arch', module_arch, all_gene_dim=gene_dim, use_cossim=config['trainer']['use_cossim'], 
                                     train_visual=config['trainer']['train_visual'], study=config['data_loader']['args']['study'], ablation_k=config['trainer']['ablation_k'])
            # logger.info(model1)
            if config['trainer']['test_ig']:    # for IG 
                model2 = SNN_IG(all_gene_dim=gene_dim)
            else:
                model2 = SNN(all_gene_dim=gene_dim)    
            model3 = AggrLayer()

            # prepare for (multi-device) GPU training
            device, device_ids = prepare_device(config['n_gpu'])
            model1 = model1.to(device)
            model2 = model2.to(device)
            model3 = model3.to(device)
            models = [model1, model2, model3]

        elif config['name'] == 'ABMIL' or config['name'] =='TransMIL'or config['name'] =='DSMIL' or config['name'] =='MamMIL' or config['name'] == 'CSMIL':  # single wsi modal
            model = config.init_obj('arch', module_arch)
            device, device_ids = prepare_device(config['n_gpu'])
            model = model.to(device)
            
        elif config['data_loader']['args']['apply_sig'] == True:    # multi wsi+gene modal 
            if config['data_loader']['args']['study'] == 'LUAD':
                omic_sizes =  [89,334,534,471,1510,482] # [99, 353, 568, 505, 1560, 495]
            elif config['data_loader']['args']['study'] == 'BRCA':
                omic_sizes = [91, 353, 553, 480, 1566, 480]
            elif config['data_loader']['args']['study'] == 'COAD':
                omic_sizes = [75, 295, 466, 402, 1351, 408]
            elif config['data_loader']['args']['study'] == 'STAD':
                omic_sizes = [75, 295, 466, 402, 1355, 416]
            elif config['data_loader']['args']['study'] == 'UCEC':
                omic_sizes = [58, 226, 361, 204, 853, 96]
            elif config['data_loader']['args']['study'] == 'KIRC':
                omic_sizes = [75, 295, 466, 402, 1353, 414]
            model = config.init_obj('arch', module_arch, omic_sizes=omic_sizes, input_dim=768)
            device, device_ids = prepare_device(config['n_gpu'])
            model = model.to(device)
        
        else:
            pass    

        criterion = importlib.import_module('utils.loss')
        criterion = getattr(criterion, config['loss'])
        metrics = [getattr(module_metric, met) for met in config['metrics']]

        trainable_params = None
        if config['name'][0:4] == 'MSAD':
            optimizer = optim.Adam( [
                                        {'params': model1.parameters()},
                                        {'params': model2.parameters()},
                                        {'params': model3.parameters()}
                                    ], lr=config['optimizer']['args']['lr'], weight_decay=config['optimizer']['args']['weight_decay'])
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['optimizer']['args']['lr_gamma'], last_epoch=-1, verbose=False)
        else:
            optimizer = optim.Adam( model.parameters(), lr=config['optimizer']['args']['lr'], weight_decay=config['optimizer']['args']['weight_decay'])
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['optimizer']['args']['lr_gamma'], last_epoch=-1, verbose=False)
        
        if config['name'][0:4] == 'MSAD':
            trainer = SurvTrainer(models, criterion, metrics, optimizer,
                            config=config,
                            device=device,
                            data_loader=train_dataloader,
                            valid_data_loader=val_dataloader,
                            test_data_loader=test_dataloader,
                            fold=i,
                            lr_scheduler=lr_scheduler)
        elif config['name'] == 'ABMIL' or config['name'] =='TransMIL' or config['name'] =='MamMIL':  # single wsi modal
            trainer = SurvTrainer_WSI(model,criterion, metrics, optimizer,
                            config=config,
                            device=device,
                            data_loader=train_dataloader,
                            valid_data_loader=val_dataloader,
                            test_data_loader=test_dataloader,
                            fold=i,
                            lr_scheduler=lr_scheduler)
        elif config['name'] =='DSMIL':  # single wsi modal
            trainer = SurvTrainer_DSMIL(model,criterion, metrics, optimizer,
                            config=config,
                            device=device,
                            data_loader=train_dataloader,
                            valid_data_loader=val_dataloader,
                            test_data_loader=test_dataloader,
                            fold=i,
                            lr_scheduler=lr_scheduler)
        elif config['name'] == 'MCAT' or config['name'] =='MoCAT' or config['name'] =='Survpath':   # WSI + gene
            trainer = SurvTrainer_MCAT(model, criterion, metrics, optimizer,
                            config=config,
                            device=device,
                            data_loader=train_dataloader,
                            valid_data_loader=val_dataloader,
                            test_data_loader=test_dataloader,
                            fold=i,
                            lr_scheduler=lr_scheduler)
        elif config['name'] == 'HiMT':  
            trainer = SurvTrainer_HiMT(model, criterion, metrics, optimizer,
                            config=config,
                            device=device,
                            data_loader=train_dataloader,
                            valid_data_loader=val_dataloader,
                            test_data_loader=test_dataloader,
                            fold=i,
                            lr_scheduler=lr_scheduler)
        elif config['name'] == 'CSMIL':  
            trainer = SurvTrainer_CSMIL(model, criterion, metrics, optimizer,
                            config=config,
                            device=device,
                            data_loader=train_dataloader,
                            valid_data_loader=val_dataloader,
                            test_data_loader=test_dataloader,
                            fold=i,
                            lr_scheduler=lr_scheduler)
             
        elif config['trainer']['ablation_single_gene'] == True:  # single gene modal
            pass
        # if not config['trainer']['test_phase']:
        #     trainer.train(fold=i)
        # else:
            # cindex_aggr, cindex_mm, cindex_gene = trainer.test(i)
        trainer.train(fold=i)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('--study', default='UCEC', type=str,
                      help='study name')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--experiment', default=None, type=str,
                      help='experiment name')
    args.add_argument('--use_trust', action='store_true', default=False, help='是否使用可信surv') 
    args.add_argument('--use_updated_genevec', action='store_true', default=False, help='是否使用更新后的genevec')
    args.add_argument('--use_cossim', action='store_true', default=False, help='是否使用余弦相似度')
    args.add_argument('--use_geneexp', action='store_true', default=False, help='是否基因表达(验证vec是否有用)')
    args.add_argument('--cossim_w',type=float, default=5e-4, help='余弦相似度权重 (default: 0.0005)')
    args.add_argument('--ce_w',type=float, default=5e-1, help='可信loss权重 (default: 0.5)')
    args.add_argument('--weight_decay',type=float, default=1e-4, help='权重递减')
    args.add_argument('--lr_gamma',type=float, default=0.9, help='学习率衰减参数')
    args.add_argument('--fold', type=int, default=4, help='Number of folds (default: 4)')
    args.add_argument('--gc', type=int, default=1, help='gc value')
    
    args.add_argument('--test_phase', action='store_true', default=False, help='是否为测试阶段')
    args.add_argument('--test_ig', action='store_true', default=False, help='是否为测试IG阶段')
    args.add_argument('--test_KM', action='store_true', default=False, help='是否为KM绘制阶段')
    args.add_argument('--train_visual', action='store_true', default=False, help='是否为可视化基因阶段')
    args.add_argument('--ablation_k', type=int, default=6, help='关键patch进行消融实验')
    
    #args.add_argument('--lr', type=float, default=5e-5, help='Learning rate (default: 0.00005)')
    
    args.add_argument('--ablation_reg', action='store_true', default=False, help='消融;是否正则') 
    args.add_argument('--ablation_geneexp', action='store_true', default=False, help='消融;用离散基因表达')
    args.add_argument('--ablation_baseline', action='store_true', default=False, help='消融;用baseline,基因使用AAAI的方法直接输入')
    args.add_argument('--ablation_single_wsi', action='store_true', default=False, help='消融;用baseline,基因使用AAAI的方法直接输入')
    args.add_argument('--ablation_single_gene', action='store_true', default=False, help='消融;用baseline,基因使用AAAI的方法直接输入')

    args.add_argument('--test_model_path', type=str, default='/test', help='测试模型地址')
    args.add_argument('--test_save_path', type=str, default='/test', help='测试结果地址')
    
    
    
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    
    # import os
    # config['trainer']['save_dir'] = os.path.join(config['trainer']['save_dir'], 'gc'+str(config['trainer']['gc']))
    main(config)
