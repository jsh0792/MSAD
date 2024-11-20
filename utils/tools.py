from pathlib import Path
import json
import torch
from collections import OrderedDict
from itertools import repeat
import pandas as pd
import numpy as np

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset)) 
    print(dataset.slide_cls_ids)                                          
    weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
    weight = [0] * int(N)                                           
    for idx in range(len(dataset)):   
        y = dataset.getlabel(idx)                        
        weight[idx] = weight_per_class[y]                                  

    return torch.DoubleTensor(weight)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, warmup=5, patience=15, stop_epoch=20, verbose=False, logger=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.logger = logger

    def __call__(self, epoch, val_loss, models, ckpt_name = 'checkpoint.pt'):

        score = val_loss

        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, models, ckpt_name, epoch)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.save_checkpoint(val_loss, models, ckpt_name, epoch)
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self, val_loss, models, ckpt_name, epoch):
        '''Saves model when validation loss decrease.'''
        self.logger.info(f'C index increse ({self.best_score:.6f} --> {val_loss:.6f}).  Saving model ... at epoch {epoch}')
        if len(models) == 1:
            torch.save(models[0].state_dict(), ckpt_name + '_mm.pt')
        else:
            torch.save(models[0].state_dict(), ckpt_name + '_mm.pt')
            torch.save(models[1].state_dict(), ckpt_name + '_gene.pt')
            torch.save(models[2].state_dict(), ckpt_name + '_aggr.pt')


def get_exp_name(args):
    exp_name = []
    if args.use_trust:
        exp_name.append('trust')
    if args.use_cossim:
        exp_name.append('cos')
    if args.use_geneexp:
        exp_name.append('geneexp')

    exp_name.append('lr%s' % format(args.lr, '.0e'))
    exp_name.append('lrGAMMA%s' % format(args.lr_gamma, '.2e'))
    if args.use_cossim:
        exp_name.append('cos%s' % format(args.cossim_w, '.0e'))
    if args.use_trust:
        exp_name.append('ce%s' % format(args.ce_w, '.0e'))
    exp_name.append('weightdecay%s' % format(args.weight_decay, '.0e'))

    exp_name = '_'.join(exp_name)
    return exp_name