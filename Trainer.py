import numpy as np
import torch
from torchvision.utils import make_grid
from BaseClass.BaseTrainer import BaseTrainer
from utils.tools import inf_loop, MetricTracker
from torch.utils.tensorboard import SummaryWriter
from utils.tools import EarlyStopping
import os
from tqdm import tqdm
from utils.loss import rankloss, ce_loss, DS_Combin

class SurvTrainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device, fold,
                 data_loader, valid_data_loader=None, test_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        self.criterion = criterion()
        self.lr_scheduler = lr_scheduler
        self.ablation_reg = config['trainer']['ablation_reg']
        self.ablation_geneexp = config['trainer']['ablation_geneexp']
        self.use_trust = config['trainer']['use_trust']
        self.use_cossim = config['trainer']['use_cossim'] 
        self.ablation_baseline = config['trainer']['ablation_baseline']
        self.test_phase = config['trainer']['test_phase']
        self.cossim_w = config['trainer']['cossim_w']
        self.ce_w = config['trainer']['ce_w']
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.writer = SummaryWriter(config['trainer']['save_dir'] + '/{}'.format(fold), flush_secs=15)   
        #self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        #self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch, early_stopping, fold):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model1.train()
        self.model2.train()
        self.model3.train()
        all_risk_scores = np.zeros((len(self.data_loader)))
        all_censorships = np.zeros((len(self.data_loader)))
        all_event_times = np.zeros((len(self.data_loader)))

        all_risk_scores_mm = np.zeros((len(self.data_loader)))
        all_censorships_mm = np.zeros((len(self.data_loader)))
        all_event_times_mm = np.zeros((len(self.data_loader)))

        all_risk_scores_gene = np.zeros((len(self.data_loader)))
        all_censorships_gene = np.zeros((len(self.data_loader)))
        all_event_times_gene = np.zeros((len(self.data_loader)))

        self.rankloss_acc_step = 6
        train_loss_surv, train_loss = 0., 0.
        rl_log, ce_gene_log, ce_mm_log, ce_aggr_log, ce_log = 0., 0., 0., 0., 0.
        cossim_log = 0.

        S_aggr_rl = []
        label_rl = []
        c_rl = []
        event_time_rl = []
        #for batch_idx, (data, target) in enumerate(self.data_loader):
        for batch_idx, (data_WSI_5x, data_WSI_10x, data_WSI_20x, gene, gene_vec, label, event_time, c, case_id) in enumerate(tqdm(self.data_loader)):
            #data, target = data.to(self.device), target.to(self.device)
            data_WSI_5x = data_WSI_5x.to(self.device)
            data_WSI_10x = data_WSI_10x.to(self.device)
            data_WSI_20x = data_WSI_20x.to(self.device)
            gene = gene.type(torch.FloatTensor).to(self.device)
            gene_vec = gene_vec.type(torch.FloatTensor).to(self.device)
            label = label.type(torch.LongTensor).to(self.device)
            c = c.type(torch.FloatTensor).to(self.device)
            
            patient_gene_vec = []
            for expression, vec in zip(gene, gene_vec):
                patient_gene_vec.append(expression * vec)
            patient_gene_vec = torch.cat(patient_gene_vec).view(-1, 200)
            
            if not self.ablation_geneexp:            
                hazards_mm, S_mm, Y_hat, mm_embedding, gene_5x_loss, gene_10x_loss, gene_20x_loss, updated_gene_vec = self.model1(x_path_5x=data_WSI_5x, x_path_10x=data_WSI_10x, 
                                                                                    x_path_20x=data_WSI_20x, gene_vec=patient_gene_vec, ablation_geneexp=self.ablation_geneexp, case_id=case_id[0])
            else:
                hazards_mm, S_mm, Y_hat, mm_embedding, gene_5x_loss, gene_10x_loss, gene_20x_loss, updated_gene_vec = self.model1(x_path_5x=data_WSI_5x, x_path_10x=data_WSI_10x,
                                                                                    x_path_20x=data_WSI_20x, gene_vec=gene, ablation_geneexp=self.ablation_geneexp, case_id=case_id[0])
            if not self.ablation_geneexp: 
                hazards_gene, S_gene, Y_hat, gene_embedding  = self.model2(gene_vec=updated_gene_vec, ablation_geneexp=self.ablation_geneexp)
            else:
                hazards_gene, S_gene, Y_hat, gene_embedding  = self.model2(gene_vec=gene, ablation_geneexp=self.ablation_geneexp)
            hazards_aggr, S_aggr, Y_hat  = self.model3(mm_embedding=mm_embedding, gene_embedding=gene_embedding)

            survloss_gene = self.criterion(hazards=hazards_mm, S=S_mm, Y=label, c=c)
            survloss_mm = self.criterion(hazards=hazards_gene, S=S_gene, Y=label, c=c)
            survloss_aggr = self.criterion(hazards=hazards_aggr, S=S_aggr, Y=label, c=c)

            ### trusted survival
            if self.use_trust:
                evidence_1 = hazards_mm
                evidence_2 = hazards_gene
                censorship = c
                class_num = 4
                ce_mm = ce_loss(evidence_1.squeeze(), class_num, epoch, 10, 100, censorship, label, self.device)
                ce_gene = ce_loss(evidence_2.squeeze(), class_num, epoch, 10, 100, censorship, label, self.device)

                alpha_1 = evidence_1 + 1
                alpha_2 = evidence_2 + 1
                alpha_list = [alpha_1, alpha_2]
                alpha_aggr = DS_Combin(alpha_list, class_num)

                ce_aggr = ce_loss((alpha_aggr-1).squeeze(), class_num, epoch, 10, 100, censorship, label, self.device)
                if self.use_cossim:
                    survloss = survloss_gene + survloss_mm + survloss_aggr + self.ce_w * (ce_mm + ce_gene + ce_aggr) + self.cossim_w * (gene_5x_loss + gene_10x_loss + gene_20x_loss)
                    #survloss = self.ce_w * (ce_mm + ce_gene + ce_aggr) + self.cossim_w * (gene_5x_loss + gene_10x_loss + gene_20x_loss)
                    cossim_log += (gene_5x_loss.item() + gene_10x_loss.item() + gene_20x_loss.item())
                else:
                    survloss = survloss_gene + survloss_mm + survloss_aggr + self.ce_w * (ce_mm + ce_gene + ce_aggr)
                ce_log += (ce_mm.item() + ce_gene.item() + ce_aggr.item())
                ce_gene_log += ce_gene.item()
                ce_mm_log += ce_mm.item()
                ce_aggr_log += ce_aggr.item()
            elif self.use_cossim:
                survloss = survloss_gene + survloss_mm + survloss_aggr + (gene_5x_loss + gene_10x_loss + gene_20x_loss) * self.cossim_w
                # survloss = survloss_gene + survloss_mm + survloss_aggr
                # cos_loss = (gene_5x_loss + gene_10x_loss + gene_20x_loss) * self.cossim_w
                cossim_log += (gene_5x_loss.item() + gene_10x_loss.item() + gene_20x_loss.item())
            else:
                if not self.ablation_reg:
                    survloss = survloss_gene + survloss_mm + survloss_aggr
                else:
                    reg_norm = sum(p.pow(2.0).sum() for p in self.model1.parameters())
                    survloss = survloss_gene + survloss_mm + survloss_aggr + 0.00000001 * reg_norm
            train_loss_surv += (survloss_gene.item() + survloss_mm.item() + survloss_aggr.item())
            train_loss += survloss.item()
            
            survloss.backward(retain_graph=True)

            ### 排序损失 choose from the two
            # if ((batch_idx + 1) % self.rankloss_acc_step) != 0:
            #     S_aggr_rl.append(S_aggr)
            #     label_rl.append(label)
            #     c_rl.append(c)
            #     event_time_rl.append(event_time)
            # else:
            #     S_aggr_rl = torch.cat(S_aggr_rl).reshape(-1, 4)
            #     label_rl = torch.cat(label_rl).reshape(-1, 1)
            #     rl = rankloss(S_aggr_rl, label_rl, c_rl, event_time_rl)
            #     train_loss += rl.item()
            #     rl_log += rl.item()
            #     rl.backward()
            #     self.optimizer.step()
            #     self.optimizer.zero_grad()
            #     S_aggr_rl = []
            #     label_rl = []
            #     c_rl = []
            #     event_time_rl = []
            
            ### without ranking loss..
            if (batch_idx + 1) % 32 == 0: 
                self.optimizer.step()
                self.optimizer.zero_grad()


            risk = -torch.sum(S_aggr, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.item()
            all_event_times[batch_idx] = event_time

            risk = -torch.sum(S_mm, dim=1).detach().cpu().numpy()
            all_risk_scores_mm[batch_idx] = risk
            all_censorships_mm[batch_idx] = c.item()
            all_event_times_mm[batch_idx] = event_time

            risk = -torch.sum(S_gene, dim=1).detach().cpu().numpy()
            all_risk_scores_gene[batch_idx] = risk
            all_censorships_gene[batch_idx] = c.item()
            all_event_times_gene[batch_idx] = event_time

        train_loss_surv /= len(self.data_loader)
        train_loss /= len(self.data_loader)
        rl_log /= len(self.data_loader)
        ce_gene_log /= len(self.data_loader)
        ce_mm_log /= len(self.data_loader)
        ce_aggr_log /= len(self.data_loader)
        ce_log /= len(self.data_loader)
        cossim_log /= len(self.data_loader)
        cindex_aggr = self.metric_ftns[0](all_censorships, all_event_times, all_risk_scores)
        cindex_mm = self.metric_ftns[0](all_censorships_mm, all_event_times_mm, all_risk_scores_mm)
        cindex_gene = self.metric_ftns[0](all_censorships_gene, all_event_times_gene, all_risk_scores_gene)

        self.writer.add_scalar('train/loss_surv', train_loss_surv, epoch)
        self.writer.add_scalar('train/loss', train_loss, epoch)
        self.writer.add_scalar('train/rl_log', rl_log, epoch)
        self.writer.add_scalar('train/ce_gene_log', ce_gene_log, epoch)
        self.writer.add_scalar('train/ce_mm_log', ce_mm_log, epoch)
        self.writer.add_scalar('train/ce_aggr_log', ce_aggr_log, epoch)
        self.writer.add_scalar('train/ce_log', ce_log, epoch)
        self.writer.add_scalar('train/cossim_log', cossim_log, epoch)
        self.writer.add_scalar('train/c_index', cindex_aggr, epoch)
        self.writer.add_scalar('train/c_index_mm', cindex_mm, epoch)
        self.writer.add_scalar('train/c_index_gene', cindex_gene, epoch)

        val_log = self._valid_epoch(epoch, early_stopping, fold)


    def _valid_epoch(self, epoch, early_stopping, fold):
        self.model1.eval()
        self.model2.eval()
        self.model3.eval()
        all_risk_scores = np.zeros((len(self.valid_data_loader)))
        all_censorships = np.zeros((len(self.valid_data_loader)))
        all_event_times = np.zeros((len(self.valid_data_loader)))

        all_risk_scores_mm = np.zeros((len(self.valid_data_loader)))
        all_censorships_mm = np.zeros((len(self.valid_data_loader)))
        all_event_times_mm = np.zeros((len(self.valid_data_loader)))

        all_risk_scores_gene = np.zeros((len(self.valid_data_loader)))
        all_censorships_gene = np.zeros((len(self.valid_data_loader)))
        all_event_times_gene = np.zeros((len(self.valid_data_loader)))
        val_loss_surv, val_loss = 0., 0.
        rl_log, ce_gene_log, ce_mm_log, ce_aggr_log, ce_log = 0., 0., 0., 0., 0.
        cossim_log = 0.
        S_aggr_rl = []
        label_rl = []
        c_rl = []
        event_time_rl = []
        #for batch_idx, (data, target) in enumerate(self.data_loader):
        for batch_idx, (data_WSI_5x, data_WSI_10x, data_WSI_20x, gene, gene_vec, label, event_time, c, case_id) in enumerate(tqdm(self.valid_data_loader)):
            data_WSI_5x = data_WSI_5x.to(self.device)
            data_WSI_10x = data_WSI_10x.to(self.device)
            data_WSI_20x = data_WSI_20x.to(self.device)
            gene = gene.type(torch.FloatTensor).to(self.device)
            gene_vec = gene_vec.type(torch.FloatTensor).to(self.device)
            label = label.type(torch.LongTensor).to(self.device)
            c = c.type(torch.FloatTensor).to(self.device)

            patient_gene_vec = []
            for expression, vec in zip(gene, gene_vec):
                patient_gene_vec.append(expression * vec)
            patient_gene_vec = torch.cat(patient_gene_vec).view(-1, 200)

            if not self.ablation_geneexp:            
                hazards_mm, S_mm, Y_hat, mm_embedding, gene_5x_loss, gene_10x_loss, gene_20x_loss, updated_gene_vec = self.model1(x_path_5x=data_WSI_5x, x_path_10x=data_WSI_10x,
                                                                                    x_path_20x=data_WSI_20x, gene_vec=patient_gene_vec, ablation_geneexp=self.ablation_geneexp, case_id=case_id[0])
            else:
                hazards_mm, S_mm, Y_hat, mm_embedding, gene_5x_loss, gene_10x_loss, gene_20x_loss, updated_gene_vec = self.model1(x_path_5x=data_WSI_5x, x_path_10x=data_WSI_10x,
                                                                                    x_path_20x=data_WSI_20x, gene_vec=gene, ablation_geneexp=self.ablation_geneexp, case_id=case_id[0])
            if not self.ablation_geneexp: 
                hazards_gene, S_gene, Y_hat, gene_embedding  = self.model2(gene_vec=updated_gene_vec, ablation_geneexp=self.ablation_geneexp)
            else:
                hazards_gene, S_gene, Y_hat, gene_embedding  = self.model2(gene_vec=gene, ablation_geneexp=self.ablation_geneexp)
            hazards_aggr, S_aggr, Y_hat  = self.model3(mm_embedding=mm_embedding, gene_embedding=gene_embedding)

            survloss_gene = self.criterion(hazards=hazards_mm, S=S_mm, Y=label, c=c)
            survloss_mm = self.criterion(hazards=hazards_gene, S=S_gene, Y=label, c=c)
            survloss_aggr = self.criterion(hazards=hazards_aggr, S=S_aggr, Y=label, c=c)

            ### trusted survival
            if self.use_trust:
                evidence_1 = hazards_mm
                evidence_2 = hazards_gene
                censorship = c
                class_num = 4
                ce_mm = ce_loss(evidence_1.squeeze(), class_num, 5, 10, 100, censorship, label, self.device)
                ce_gene = ce_loss(evidence_2.squeeze(), class_num, 5, 10, 100, censorship, label, self.device)

                alpha_1 = evidence_1 + 1
                alpha_2 = evidence_2 + 1
                alpha_list = [alpha_1, alpha_2]
                alpha_aggr = DS_Combin(alpha_list, class_num)

                ce_aggr = ce_loss((alpha_aggr-1).squeeze(), class_num, 5, 10, 100, censorship, label, self.device)
                if self.use_cossim:
                    survloss = survloss_gene + survloss_mm + survloss_aggr + self.ce_w * (ce_mm + ce_gene + ce_aggr) + self.cossim_w * (gene_5x_loss + gene_10x_loss + gene_20x_loss)
                    cossim_log += (gene_5x_loss.item() + gene_10x_loss.item() + gene_20x_loss.item())
                else:
                    survloss = survloss_gene + survloss_mm + survloss_aggr + self.ce_w * (ce_mm + ce_gene + ce_aggr)
                ce_log += (ce_mm.item() + ce_gene.item() + ce_aggr.item())
                ce_gene_log += ce_gene.item()
                ce_mm_log += ce_mm.item()
                ce_aggr_log += ce_aggr.item()
            elif self.use_cossim:
                survloss = survloss_gene + survloss_mm + survloss_aggr + (gene_5x_loss + gene_10x_loss + gene_20x_loss) * self.cossim_w
                # survloss = survloss_gene + survloss_mm + survloss_aggr
                # cos_loss = (gene_5x_loss + gene_10x_loss + gene_20x_loss) * self.cossim_w
                cossim_log += (gene_5x_loss.item() + gene_10x_loss.item() + gene_20x_loss.item())
            else:
                survloss = survloss_gene + survloss_mm + survloss_aggr
            val_loss_surv += (survloss_gene.item() + survloss_mm.item() + survloss_aggr.item())
            val_loss += survloss.item()
            
            #survloss.backward(retain_graph=True)

            ### 排序损失
            if ((batch_idx + 1) % self.rankloss_acc_step) != 0:
                S_aggr_rl.append(S_aggr)
                label_rl.append(label)
                c_rl.append(c)
                event_time_rl.append(event_time)
            else:
                S_aggr_rl = torch.cat(S_aggr_rl).reshape(-1, 4)
                label_rl = torch.cat(label_rl).reshape(-1, 1)
                rl = rankloss(S_aggr_rl, label_rl, c_rl, event_time_rl)
                val_loss += rl.item()
                rl_log += rl.item()
                #rl.backward()
                # self.optimizer.step()
                # self.optimizer.zero_grad()
                S_aggr_rl = []
                label_rl = []
                c_rl = []
                event_time_rl = []

            risk = -torch.sum(S_aggr, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.item()
            all_event_times[batch_idx] = event_time

            risk = -torch.sum(S_mm, dim=1).detach().cpu().numpy()
            all_risk_scores_mm[batch_idx] = risk
            all_censorships_mm[batch_idx] = c.item()
            all_event_times_mm[batch_idx] = event_time

            risk = -torch.sum(S_gene, dim=1).detach().cpu().numpy()
            all_risk_scores_gene[batch_idx] = risk
            all_censorships_gene[batch_idx] = c.item()
            all_event_times_gene[batch_idx] = event_time

            torch.cuda.empty_cache()

        val_loss_surv /= len(self.data_loader)
        val_loss /= len(self.data_loader)
        rl_log /= len(self.data_loader)
        ce_gene_log /= len(self.data_loader)
        ce_mm_log /= len(self.data_loader)
        ce_aggr_log /= len(self.data_loader)
        ce_log /= len(self.data_loader)
        cossim_log /= len(self.data_loader)
        cindex_aggr = self.metric_ftns[0](all_censorships, all_event_times, all_risk_scores)
        cindex_mm = self.metric_ftns[0](all_censorships_mm, all_event_times_mm, all_risk_scores_mm)
        cindex_gene = self.metric_ftns[0](all_censorships_gene, all_event_times_gene, all_risk_scores_gene)

        self.writer.add_scalar('val/loss_surv', val_loss_surv, epoch)
        self.writer.add_scalar('val/loss', val_loss, epoch)
        self.writer.add_scalar('val/rl_log', rl_log, epoch)
        self.writer.add_scalar('val/ce_gene_log', ce_gene_log, epoch)
        self.writer.add_scalar('val/ce_mm_log', ce_mm_log, epoch)
        self.writer.add_scalar('val/ce_aggr_log', ce_aggr_log, epoch)
        self.writer.add_scalar('val/ce_log', ce_log, epoch)
        self.writer.add_scalar('val/cossim_log', cossim_log, epoch)
        self.writer.add_scalar('val/c_index', cindex_aggr, epoch)
        self.writer.add_scalar('val/c_index_mm', cindex_mm, epoch)
        self.writer.add_scalar('val/c_index_gene', cindex_gene, epoch)

        early_stopping(epoch, cindex_aggr, [self.model1, self.model2, self.model3], ckpt_name=os.path.join(self.config['trainer']['save_dir'], str(fold), "s_{}_bestcindex".format(fold)))

    def test(self, fold):
        if self.config['trainer']['test_phase']:
            self.model1.load_state_dict(torch.load(os.path.join(self.config['trainer']['test_model_path'], str(fold), "s_{}_bestcindex_mm.pt".format(fold))))
            self.model2.load_state_dict(torch.load(os.path.join(self.config['trainer']['test_model_path'], str(fold), "s_{}_bestcindex_gene.pt".format(fold))))
            self.model3.load_state_dict(torch.load(os.path.join(self.config['trainer']['test_model_path'], str(fold), "s_{}_bestcindex_aggr.pt".format(fold))))
        else:
            self.model1.load_state_dict(torch.load(os.path.join(self.config['trainer']['save_dir'], str(fold), "s_{}_bestcindex_mm.pt".format(fold))))
            self.model2.load_state_dict(torch.load(os.path.join(self.config['trainer']['save_dir'], str(fold), "s_{}_bestcindex_gene.pt".format(fold))))
            self.model3.load_state_dict(torch.load(os.path.join(self.config['trainer']['save_dir'], str(fold), "s_{}_bestcindex_aggr.pt".format(fold))))

        self.model1.eval()
        self.model2.eval()
        self.model3.eval()
        all_risk_scores = np.zeros((len(self.test_data_loader)))
        all_censorships = np.zeros((len(self.test_data_loader)))
        all_event_times = np.zeros((len(self.test_data_loader)))

        all_risk_scores_mm = np.zeros((len(self.test_data_loader)))
        all_censorships_mm = np.zeros((len(self.test_data_loader)))
        all_event_times_mm = np.zeros((len(self.test_data_loader)))

        all_risk_scores_gene = np.zeros((len(self.test_data_loader)))
        all_censorships_gene = np.zeros((len(self.test_data_loader)))
        all_event_times_gene = np.zeros((len(self.test_data_loader)))
        #for batch_idx, (data, target) in enumerate(self.data_loader):
        for batch_idx, (data_WSI_5x, data_WSI_10x, data_WSI_20x, gene, gene_vec, label, event_time, c, case_id) in enumerate(tqdm(self.test_data_loader)):
            #data, target = data.to(self.device), target.to(self.device)
            data_WSI_5x = data_WSI_5x.to(self.device)
            data_WSI_10x = data_WSI_10x.to(self.device)
            data_WSI_20x = data_WSI_20x.to(self.device)
            gene = gene.type(torch.FloatTensor).to(self.device)
            gene_vec = gene_vec.type(torch.FloatTensor).to(self.device)
            label = label.type(torch.LongTensor).to(self.device)
            c = c.type(torch.FloatTensor).to(self.device)

            patient_gene_vec = []
            for expression, vec in zip(gene, gene_vec):
                patient_gene_vec.append(expression * vec)
            patient_gene_vec = torch.cat(patient_gene_vec).reshape(-1, 200)

            with torch.no_grad():
                if not self.ablation_geneexp:            
                    hazards_mm, S_mm, Y_hat, mm_embedding, gene_5x_loss, gene_10x_loss, gene_20x_loss, updated_gene_vec = self.model1(x_path_5x=data_WSI_5x, x_path_10x=data_WSI_10x, 
                                                                    x_path_20x=data_WSI_20x, gene_vec=patient_gene_vec, ablation_geneexp=self.ablation_geneexp, case_id=case_id[0])
                else:
                    hazards_mm, S_mm, Y_hat, mm_embedding, gene_5x_loss, gene_10x_loss, gene_20x_loss, updated_gene_vec = self.model1(x_path_5x=data_WSI_5x, x_path_10x=data_WSI_10x,
                                                                    x_path_20x=data_WSI_20x, gene_vec=gene, ablation_geneexp=self.ablation_geneexp, case_id=case_id[0])
                if not self.ablation_geneexp: 
                    hazards_gene, S_gene, Y_hat, gene_embedding  = self.model2(gene_vec=updated_gene_vec, ablation_geneexp=self.ablation_geneexp)
                else:
                    hazards_gene, S_gene, Y_hat, gene_embedding  = self.model2(gene_vec=gene, ablation_geneexp=self.ablation_geneexp)
                hazards_aggr, S_aggr, Y_hat  = self.model3(mm_embedding=mm_embedding, gene_embedding=gene_embedding)    

            risk = -torch.sum(S_aggr, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.item()
            all_event_times[batch_idx] = event_time

            risk = -torch.sum(S_mm, dim=1).detach().cpu().numpy()
            all_risk_scores_mm[batch_idx] = risk
            all_censorships_mm[batch_idx] = c.item()
            all_event_times_mm[batch_idx] = event_time

            risk = -torch.sum(S_gene, dim=1).detach().cpu().numpy()
            all_risk_scores_gene[batch_idx] = risk
            all_censorships_gene[batch_idx] = c.item()
            all_event_times_gene[batch_idx] = event_time

        cindex_aggr = self.metric_ftns[0](all_censorships, all_event_times, all_risk_scores)
        cindex_mm = self.metric_ftns[0](all_censorships_mm, all_event_times_mm, all_risk_scores_mm)
        cindex_gene = self.metric_ftns[0](all_censorships_gene, all_event_times_gene, all_risk_scores_gene)

        np.savez(os.path.join(self.config['trainer']['save_dir'], str(fold), 'all_risk_scores.npz'), all_risk_scores)
        np.savez(os.path.join(self.config['trainer']['save_dir'], str(fold), 'all_censorships.npz'), all_censorships)
        np.savez(os.path.join(self.config['trainer']['save_dir'], str(fold), 'all_event_times.npz'), all_event_times)

        return cindex_aggr, cindex_mm, cindex_gene

    def test_ig(self, fold=0):
        dataloader = self.valid_data_loader # test dataloader
        from captum.attr import IntegratedGradients
        self.model1.load_state_dict(torch.load(os.path.join(self.config['trainer']['save_dir'], str(fold), "s_{}_bestcindex_mm.pt".format(fold))))
        self.model2.load_state_dict(torch.load(os.path.join(self.config['trainer']['save_dir'], str(fold), "s_{}_bestcindex_gene.pt".format(fold))))
        self.model3.load_state_dict(torch.load(os.path.join(self.config['trainer']['save_dir'], str(fold), "s_{}_bestcindex_aggr.pt".format(fold))))
        ig = IntegratedGradients(self.model2)
        
        for batch_idx, (data_WSI_5x, data_WSI_10x, data_WSI_20x, gene, gene_vec, label, event_time, c, case_id) in enumerate(tqdm(dataloader)):
            data_WSI_5x = data_WSI_5x.to(self.device)
            data_WSI_10x = data_WSI_10x.to(self.device)
            data_WSI_20x = data_WSI_20x.to(self.device)
            gene = gene.type(torch.FloatTensor).to(self.device)
            gene_vec = gene_vec.type(torch.FloatTensor).to(self.device)
            label = label.type(torch.LongTensor).to(self.device)
            c = c.type(torch.FloatTensor).to(self.device)
            
            patient_gene_vec = []
            for expression, vec in zip(gene, gene_vec):
                patient_gene_vec.append(expression * vec)
            patient_gene_vec = torch.cat(patient_gene_vec).view(-1, 200)
            
            if not self.ablation_geneexp:            
                hazards_mm, S_mm, Y_hat, mm_embedding, gene_5x_loss, gene_10x_loss, gene_20x_loss, updated_gene_vec = self.model1(x_path_5x=data_WSI_5x, x_path_10x=data_WSI_10x, x_path_20x=data_WSI_20x, gene_vec=patient_gene_vec, ablation_geneexp=self.ablation_geneexp, case_id=case_id[0])
                ig = IntegratedGradients(self.model2)
                attributions, delta = ig.attribute(updated_gene_vec, n_steps=1,target=3, return_convergence_delta=True)
                
                study = self.config['data_loader']['args']['study'] 
                mean_attributions = attributions.mean(dim=1)  # 计算每行的均值
                save_path = os.path.join('/data/jsh/dfri_final/view/IG', study, str(case_id[0])+'_mean_attributions.pt' )
                torch.save(mean_attributions, save_path)
                
                import pandas as pd
                mean_attributions = torch.load(save_path).cpu().detach()
                mean_attributions = mean_attributions.squeeze()  # 转换为一维张量
                mean_attributions_series = pd.Series(mean_attributions.numpy())  # 转换为 Pandas Series
                csv_file_path = '/data/jsh/dfri_final/view/IG/{}/keys.csv'.format(study)
                df = pd.read_csv(csv_file_path)
                df[case_id[0]] = mean_attributions_series
                df.to_csv(csv_file_path, index=False)
            
            else:
                hazards_mm, S_mm, Y_hat, mm_embedding, gene_5x_loss, gene_10x_loss, gene_20x_loss, updated_gene_vec = self.model1(x_path_5x=data_WSI_5x, x_path_10x=data_WSI_10x, x_path_20x=data_WSI_20x, gene_vec=patient_gene_vec, ablation_geneexp=self.ablation_geneexp, case_id=case_id[0])
                hazards_gene, S_gene, Y_hat, gene_embedding  = self.model2(gene_vec=gene)
                baseline = torch.zeros(gene_vec.shape).cuda()
                attributions, delta = ig.attribute(gene, target=label, return_convergence_delta=True)
            
            print('IG Attributions:', attributions.shape)
            print('Convergence Delta:', delta)        

    def view_gene(self, fold=0):
        dataloader = self.data_loader
        self.model1.load_state_dict(torch.load(os.path.join(self.config['trainer']['save_dir'], str(fold), "s_{}_bestcindex_mm.pt".format(fold))))
        self.model2.load_state_dict(torch.load(os.path.join(self.config['trainer']['save_dir'], str(fold), "s_{}_bestcindex_gene.pt".format(fold))))
        self.model3.load_state_dict(torch.load(os.path.join(self.config['trainer']['save_dir'], str(fold), "s_{}_bestcindex_aggr.pt".format(fold))))
        
        for batch_idx, (data_WSI_5x, data_WSI_10x, data_WSI_20x, gene, gene_vec, label, event_time, c, case_id) in enumerate(tqdm(dataloader)):
            data_WSI_5x = data_WSI_5x.to(self.device)
            data_WSI_10x = data_WSI_10x.to(self.device)
            data_WSI_20x = data_WSI_20x.to(self.device)
            gene = gene.type(torch.FloatTensor).to(self.device)
            gene_vec = gene_vec.type(torch.FloatTensor).to(self.device)
            label = label.type(torch.LongTensor).to(self.device)
            c = c.type(torch.FloatTensor).to(self.device)
            patient_gene_vec = []
            for expression, vec in zip(gene, gene_vec):
                patient_gene_vec.append(expression * vec)
            patient_gene_vec = torch.cat(patient_gene_vec).view(-1, 200)
            if not self.ablation_geneexp:            
                hazards_mm, S_mm, Y_hat, mm_embedding, gene_5x_loss, gene_10x_loss, gene_20x_loss, updated_gene_vec = self.model1(x_path_5x=data_WSI_5x, x_path_10x=data_WSI_10x, x_path_20x=data_WSI_20x, 
                                                                                                                gene_vec=patient_gene_vec, ablation_geneexp=self.ablation_geneexp, case_id=case_id[0])
            else:
                hazards_mm, S_mm, Y_hat, mm_embedding, gene_5x_loss, gene_10x_loss, gene_20x_loss, updated_gene_vec = self.model1(x_path_5x=data_WSI_5x, x_path_10x=data_WSI_10x, x_path_20x=data_WSI_20x,
                                                                                                                gene_vec=gene, ablation_geneexp=self.ablation_geneexp, case_id=case_id[0])
            if batch_idx == 4:  # 5张
                break

    def test_KM(self, fold=0):
        
        data_loader = self.data_loader   # train_data_loader
        if self.config['trainer']['test_phase']:
            self.model1.load_state_dict(torch.load(os.path.join(self.config['trainer']['test_model_path'], str(fold), "s_{}_bestcindex_mm.pt".format(fold))))
            self.model2.load_state_dict(torch.load(os.path.join(self.config['trainer']['test_model_path'], str(fold), "s_{}_bestcindex_gene.pt".format(fold))))
            self.model3.load_state_dict(torch.load(os.path.join(self.config['trainer']['test_model_path'], str(fold), "s_{}_bestcindex_aggr.pt".format(fold))))
        else:
            self.model1.load_state_dict(torch.load(os.path.join(self.config['trainer']['save_dir'], str(fold), "s_{}_bestcindex_mm.pt".format(fold))))
            self.model2.load_state_dict(torch.load(os.path.join(self.config['trainer']['save_dir'], str(fold), "s_{}_bestcindex_gene.pt".format(fold))))
            self.model3.load_state_dict(torch.load(os.path.join(self.config['trainer']['save_dir'], str(fold), "s_{}_bestcindex_aggr.pt".format(fold))))

        self.model1.eval()
        self.model2.eval()
        self.model3.eval()
        
        gt_all_risk_scores = np.zeros((len(data_loader)))
        all_risk_scores = np.zeros((len(data_loader)))
        all_censorships = np.zeros((len(data_loader)))
        all_event_times = np.zeros((len(data_loader)))
        
        #for batch_idx, (data, target) in enumerate(self.data_loader):
        for batch_idx, (data_WSI_5x, data_WSI_10x, data_WSI_20x, gene, gene_vec, label, event_time, c, case_id) in enumerate(tqdm(data_loader)):
            #data, target = data.to(self.device), target.to(self.device)
            data_WSI_5x = data_WSI_5x.to(self.device)
            data_WSI_10x = data_WSI_10x.to(self.device)
            data_WSI_20x = data_WSI_20x.to(self.device)
            gene = gene.type(torch.FloatTensor).to(self.device)
            gene_vec = gene_vec.type(torch.FloatTensor).to(self.device)
            label = label.type(torch.LongTensor).to(self.device)
            c = c.type(torch.FloatTensor).to(self.device)

            patient_gene_vec = []
            for expression, vec in zip(gene, gene_vec):
                patient_gene_vec.append(expression * vec)
            patient_gene_vec = torch.cat(patient_gene_vec).reshape(-1, 200)

            with torch.no_grad():
                if not self.ablation_geneexp:            
                    hazards_mm, S_mm, Y_hat, mm_embedding, gene_5x_loss, gene_10x_loss, gene_20x_loss, updated_gene_vec = self.model1(x_path_5x=data_WSI_5x, x_path_10x=data_WSI_10x, 
                                                                    x_path_20x=data_WSI_20x, gene_vec=patient_gene_vec, ablation_geneexp=self.ablation_geneexp, case_id=case_id[0])
                else:
                    hazards_mm, S_mm, Y_hat, mm_embedding, gene_5x_loss, gene_10x_loss, gene_20x_loss, updated_gene_vec = self.model1(x_path_5x=data_WSI_5x, x_path_10x=data_WSI_10x,
                                                                    x_path_20x=data_WSI_20x, gene_vec=gene, ablation_geneexp=self.ablation_geneexp, case_id=case_id[0])
                if not self.ablation_geneexp: 
                    hazards_gene, S_gene, Y_hat, gene_embedding  = self.model2(gene_vec=updated_gene_vec, ablation_geneexp=self.ablation_geneexp)
                else:
                    hazards_gene, S_gene, Y_hat, gene_embedding  = self.model2(gene_vec=gene, ablation_geneexp=self.ablation_geneexp)
                hazards_aggr, S_aggr, Y_hat  = self.model3(mm_embedding=mm_embedding, gene_embedding=gene_embedding)    

            risk = -torch.sum(S_aggr, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.item()
            all_event_times[batch_idx] = event_time
            gt_all_risk_scores[batch_idx] = -event_time     #  ground_truth中, 目前设置risk是event_time的相反数
            
        np.savez(os.path.join(self.config['trainer']['save_dir'], str(fold), 'train_all_censorships.npz'), all_censorships)
        np.savez(os.path.join(self.config['trainer']['save_dir'], str(fold), 'train_all_event_times.npz'), all_event_times)
        np.savez(os.path.join(self.config['trainer']['save_dir'], str(fold), 'train_all_risk_scores.npz'), all_risk_scores)
        np.savez(os.path.join(self.config['trainer']['save_dir'], str(fold), 'gt_train_all_risk_scores.npz'), gt_all_risk_scores)
    
    def train(self, fold=0):
        ### gene viewing..
        if self.config['trainer']['train_visual']:  # 基因可视化阶段
            self.view_gene(fold=fold)
            return 
        
        ### IG testing..
        if self.config['trainer']['test_ig']:   # IG测试阶段
            self.test_ig(fold=fold)
            return
        
        ### KM drawing..
        if self.config['trainer']['test_KM']:
            self.test_KM(fold=fold)
            return
        
        early_stopping = EarlyStopping(warmup=5, patience=4, stop_epoch=15, verbose = True, logger=self.logger)
        if self.config['trainer']['test_phase']:
            cindex_aggr, cindex_mm, cindex_gene = self.test(fold)
            result_file = os.path.join(self.config['trainer']['test_save_dir'], 'result.txt')
            with open(result_file, 'a+') as file:
                file.write('fold {}: aggr_cindex: {:.4f} , mm_cindex: {:.4f}, gene_cindex: {:.4f}\n'.format(fold, float(cindex_aggr), float(cindex_mm), float(cindex_gene)))
            if fold == 3:
                with open(result_file, 'r') as file:
                    cindex_list = []
                    for line in file:
                        cindex_list.append(float(line[21:27]))
                with open(result_file, 'a+') as file:
                    file.write('avg_cindex: {:.4f}, std: {:.4f}'.format(np.mean(cindex_list), np.std(cindex_list)))
        else:
            for epoch in range(0, self.epochs):
                result = self._train_epoch(epoch, early_stopping, fold)
                self.lr_scheduler.step()
                if early_stopping.early_stop == True:
                    self.logger.info('fold {}: Training stop at epoch {}'.format(str(fold), epoch))
                    break
            if early_stopping.early_stop == False:
                self.logger.info('fold {}: Training stop at epoch {}'.format(str(fold), self.epochs-1))
            cindex_aggr, cindex_mm, cindex_gene = self.test(fold)
            result_file = os.path.join(self.config['trainer']['save_dir'], 'result.txt')
            with open(result_file, 'a+') as file:
                file.write('fold {}: aggr_cindex: {:.4f} , mm_cindex: {:.4f}, gene_cindex: {:.4f}\n'.format(fold, float(cindex_aggr), float(cindex_mm), float(cindex_gene)))
            if fold == 3:
                with open(result_file, 'r') as file:
                    cindex_list = []
                    for line in file:
                        cindex_list.append(float(line[21:27]))
                with open(result_file, 'a+') as file:
                    file.write('avg_cindex: {:.4f}, std: {:.4f}'.format(np.mean(cindex_list), np.std(cindex_list)))
