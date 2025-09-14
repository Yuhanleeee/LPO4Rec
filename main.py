import torch
import random
import argparse
import pickle
import os
import logging
import time
import numpy as np
import torch.backends.cudnn as cudnn
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from model_utils.pre_optimizer_utils import add_optimizer_args, get_total_steps, configure_optimizers
from dataset_utils.seq import load_data, add_datasets_args
from dataset_utils.universal_datamodule import UniversalDataModule
from ckpt_utils.universal_ckpt import UniversalCheckpoint 
from torch.utils.data._utils.collate import default_collate
from models.sasrec import SASRec
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def notail_item(datapkl, ratio):
    items = []
    for seq in datapkl:
        items += list(seq.values())[0]
    num_dict = Counter(items)
    notail_item = [i[0] for i in num_dict.most_common(int(len(num_dict)*(1-ratio)))]
    return notail_item


def notail_item_interaction(datapkl, ratio):
    items = []
    for seq in datapkl:
        items += list(seq.values())[0]
    num_dict = Counter(items)
    all_intr_nums = sum(list(num_dict.values()))
    pop_inter = int(ratio*all_intr_nums)
    notail_item = []
    count_acc = 0
    for item, num in zip(num_dict.keys(), num_dict.values()):
        count_acc += num
        if count_acc < pop_inter:
            notail_item.append(item)
        else:
            break
    return notail_item
        
        
class Collator():
    def __init__(self, args, notail_items):
        self.len_seq = args.seq_len
        
        with open('/data/' + args.datasets_name+'/asin2id.pickle', 'rb') as f:
            self.id_asin_dict = pickle.load(f)
        self.notail_item = [self.id_asin_dict[item]+1 for item in notail_items]
        self.tail_item = [self.id_asin_dict[item]+1 for item in self.id_asin_dict if item not in notail_items]
     
    def __call__(self, inputs):
        examples = []
        
        for idx_temp, input_temp in enumerate(inputs):
            example = {}
            example['userid'] = list(input_temp.keys())[0]

            input_temp = list(input_temp.values())[0]
            seqs_temp = input_temp[-(self.len_seq+1):]
            seq_temp = seqs_temp[:-1]
            label = seqs_temp[-1]
            seq_temp_id = [self.id_asin_dict[i]+1 for i in seq_temp]  ## +1 as padding is 0
            label_id = self.id_asin_dict[label] + 1
            seq_temp_id_pad = [0] * (self.len_seq - len(seq_temp_id)) + seq_temp_id
            mask_temp = [0] * (self.len_seq - len(seq_temp_id)) + len(seq_temp_id) * [1]
            example['seq_id'] =  seq_temp_id_pad
            example['mask_seq'] = mask_temp 
            example['label_id'] = label_id
           
            if label_id not in self.notail_item:
                example['cold_item_flag'] = 1
                example['plenty'] = 1.0
            else:
                example['cold_item_flag'] = 0
                example['plenty'] = 0.0
            examples.append(example)
        
        return default_collate(examples)
               

def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def report_memory(name):
    """Simple GPU memory report."""
    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(torch.cuda.memory_allocated() / mega_bytes)
    string += ' | max allocated: {}'.format(torch.cuda.max_memory_allocated() / mega_bytes)
    string += ' | reserved: {}'.format(torch.cuda.memory_reserved() / mega_bytes)
    string += ' | max reserved: {}'.format(torch.cuda.max_memory_reserved() / mega_bytes)
    print(string)


def cal_hr(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = label == topk_predict
    hr = [hit[:, :ks[i]].sum().item()/label.size()[0] for i in range(len(ks))]
    return hr


def cal_ndcg(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = (label == topk_predict).int()
    ndcg = []
    for k in ks:
        max_dcg = dcg(torch.tensor([1] + [0] * (k-1)))
        predict_dcg = dcg(hit[:, :k])
        ndcg.append((predict_dcg/max_dcg).mean().item())
    return ndcg


def dcg(hit):
    log2 = torch.log2(torch.arange(1, hit.size()[-1] + 1) + 1).unsqueeze(0)
    rel = (hit/log2).sum(dim=-1)
    return rel


def hrs_and_ndcgs_k(scores, labels, ks):
    metrics = {}
    ndcg = cal_ndcg(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    hr = cal_hr(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    for k, ndcg_temp, hr_temp in zip(ks, ndcg, hr):
        metrics['HR@%d' % k] = hr_temp
        metrics['NDCG@%d' % k] = ndcg_temp
    return metrics  


def calculate_metrics(scores, labels, metric_ks):
    metrics = hrs_and_ndcgs_k(scores, labels, metric_ks)
    return metrics


def winsorize_rows(data, lower_percentile=5, upper_percentile=95):
    lowers = np.percentile(data, lower_percentile, axis=1, keepdims=True)
    uppers = np.percentile(data, upper_percentile, axis=1, keepdims=True)
    lowers_bc = np.broadcast_to(lowers, data.shape)
    uppers_bc = np.broadcast_to(uppers, data.shape)
    return np.where(data < lowers_bc, lowers_bc, np.where(data > uppers_bc, uppers_bc, data))


def plot_distribute(prob, tail_items):
    prob = torch.softmax(prob, dim=-1)
    prob_mean = prob.mean(1)
    prob_tail_mean = prob[:,tail_items].mean(1)
    data_plot_mean = (prob_tail_mean - prob_mean).cpu().numpy()
    
    prob_meaind = torch.median(prob, dim=-1).values
    prob_tail_meaind = torch.median(prob[:,tail_items], dim=-1).values
    data_plot_meaind = (prob_tail_meaind - prob_meaind).cpu().numpy()
    
    np.save('mean_minus_SimPO_beauty.npy', data_plot_mean)
    np.save('meaind_minus_SimPO_beauty.npy', data_plot_meaind)
    
    quit()
    # winsorized_data = winsorize_rows(prob)
    # data_plot = kurtosis(winsorized_data, bias=False)
    
    sns.histplot(data_plot_meaind, kde=True, bins=100, edgecolor='black', stat="density")
    plt.title("Combined Histogram + KDE")
    plt.xlabel("Value")
    plt.ylabel("Density/Frequency")
    plt.savefig('SASRec_meaind.png', format='png', dpi=1024)
    plt.show()
    

class TailImgRec(LightningModule):
    def add_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('Baselines Rec')
        parser.add_argument('--seq_len', type=int, default=10)
        parser.add_argument('--hidden_size', type=int, default=768)
        parser.add_argument('--attn_head', type=int, default=16)
        parser.add_argument('--n_blocks', type=int, default=1)
        parser.add_argument('--dropout', type=float, default=0.8)
        parser.add_argument('--info', type=str, default=None)
        parser.add_argument('--item_nums', type=int, default=None)
        parser.add_argument('--train_nums', type=int, default=None)
        parser.add_argument('--tau', type=float, default=0.1)
        return parent_parser
    
    def __init__(self, args, tail_items, notail_items, usernums, logger):
        super().__init__()
        
        self.tail_items = tail_items
        self.notail_items = notail_items
       
        ### ID Embedding
        self.item_emb = torch.nn.Embedding(args.item_nums+1, args.hidden_size)
        
        ### Img Embedding
        # emb = torch.load('/data/'+args.datasets_name+'/img_emb_clip_vit_large_patch14.pt')
        ### Text Embedding
        # emb = torch.load('/data/'+args.datasets_name+'/title_emb_clip_vit_large_patch14_len10.pt')
        # pad_emb = torch.mean(emb, dim=0)
        # embs = torch.cat([pad_emb.unsqueeze(0), emb], dim=0)
        # self.item_emb = torch.nn.Embedding.from_pretrained(embs)
        
        self.item_emb.weight.requires_grad = True
        
        self.rec_model= SASRec(args)
        self.loss_ce = nn.CrossEntropyLoss(reduction='none')
        self.loss_ce_raw = nn.CrossEntropyLoss()
        self.logger_save = logger
        self.tau = args.tau
        self.prob_test = []
        self.save_hyperparameters(args)
    
    def setup(self, stage) -> None:
        if stage == 'fit':
            self.total_steps = get_total_steps(self.trainer, self.hparams)
            print('Total steps: {}' .format(self.total_steps))
    
    def configure_optimizers(self):
        return configure_optimizers(self)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def gumbel_softmax_sample(self, logits, temperature=0.5):
        noise = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(noise + 1e-9) + 1e-9)
        y = logits + gumbel_noise
        return torch.nn.functional.softmax(y / temperature, dim=-1)
    
    def gumbel_topk(self, logits, k, temperature=1):
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        noisy_logits = (logits + gumbel_noise) / temperature
        topk_values, topk_indices = torch.topk(noisy_logits, k, dim=-1)
        one_hot = torch.nn.functional.one_hot(topk_indices, num_classes=logits.size(-1)).float()
        hard_one_hot = one_hot.detach() + one_hot - one_hot.detach()
        return hard_one_hot, topk_indices

    def gumbel_randomk(self, logits, k, temperature=0.5):
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-6) + 1e-6)
        noisy_logits = (gumbel_noise) / temperature
        topk_values, topk_indices = torch.topk(noisy_logits, k, dim=-1)
        one_hot = torch.nn.functional.one_hot(topk_indices, num_classes=logits.size(-1)).float()
        hard_one_hot = one_hot.detach() + one_hot - one_hot.detach()
        return hard_one_hot, topk_indices
    
    def lpo_loss(self, user_id, prod, label, tail_flag):
        k = 5
        tau =5.0
        prod = torch.softmax(prod, dim=-1)
        prod_pos = torch.gather(prod, 1, label.unsqueeze(1))
        prod_pos = torch.exp(prod_pos/tau)
        
        prod_negs = prod.clone()
        # prod_negs[torch.arange(prod_pos.shape[0]), label]=-1e8  ## remove label
        
       
        ### only consider the current step prediction
        # prod_negs_pre = prod_negs
        
        ### Sample based on probability
        # indx_negs = torch.multinomial(torch.softmax(prod_negs_pre, dim=-1), k, replacement=False)
        
        ### Sample based on TopK
        # _, indx_negs = torch.topk(prod_negs_pre, k=k)
        
        ### Randomly Sample
        # indx_negs = torch.randint(low=0, high=prod_negs.shape[1]-1, size=(prod_negs.shape[0], k)).to(prod.device)
        
        ### Randomly Sample tail item
        # indx_negs = [random.sample(self.tail_items, k=k) for ii in range(prod_negs.shape[0])]
        # indx_negs = torch.tensor(indx_negs).to(prod_pos.device)
        
        ### Randomly Sample head item
        # indx_negs = [random.sample(self.notail_items, k=k) for ii in range(prod_negs.shape[0])]
        # indx_negs = torch.tensor(indx_negs).to(prod_pos.device)
        
        ### Prob sample from head item
        prob_head = prod_negs[:, self.notail_items]
        indx_head = torch.multinomial(torch.softmax(prob_head, dim=-1), k, replacement=False)
        indx_negs = torch.tensor(self.notail_items).to(indx_head.device)[indx_head]
        
        ### TopK sample from head item
        # prob_head = prod_negs[:, self.notail_items]
        # _, indx_heads = torch.topk(prob_head, k=k)
        # indx_negs = torch.tensor(self.notail_items).to(indx_heads.device)[indx_heads]
        
        
        ### remove label from neg_indx
        while (indx_negs == label.unsqueeze(-1)).any():
            # sample_random = torch.tensor(random.choices(self.notail_items, k=indx_negs.shape[0]*indx_negs.shape[1])).reshape(indx_negs.shape[0], indx_negs.shape[1]).to(indx_negs)  ## can be put
            sample_random = torch.tensor(random.sample(self.notail_items, indx_negs.shape[0]*indx_negs.shape[1])).reshape(indx_negs.shape[0], indx_negs.shape[1]).to(indx_negs)
            indx_negs = torch.where(indx_negs == label.unsqueeze(-1), sample_random, indx_negs)
             
        ### Prob sample from tail item
        # prob_tail = prod_negs[:, self.tail_items]
        # indx_tail = torch.multinomial(torch.softmax(prob_tail, dim=-1), k, replacement=False)
        # indx_negs = torch.tensor(self.tail_items).to(indx_tail.device)[indx_tail]
        
        prod_negs_topk = torch.gather(prod, 1, indx_negs)
        neg_pos_prob = torch.sum(torch.exp(prod_negs_topk/tau), dim=-1) + prod_pos.squeeze(-1)
        loss_lpo = prod_pos.squeeze(-1)/neg_pos_prob
        
        ### Tail label loss
        # loss_lpo = tail_flag * loss_lpo
        # if torch.sum(tail_flag)>0:
        #     return torch.sum(loss_lpo)/torch.sum(tail_flag)
        # else:
        #     return 0
        
        # return loss_lpo.mean()
        return loss_lpo

    def lpo_loss_gumbel(self, user_id, prod, label, tail_flag):
        k = 5
        tau = 2.0
        prod = torch.softmax(prod, dim=-1)
        prod_pos = torch.gather(prod, 1, label.unsqueeze(1))
        prod_pos = torch.exp(prod_pos/tau)
        
        prob_negs = prod.clone()
        ## label eliminate
        mask_lable = torch.ones_like(prob_negs, dtype=torch.bool).to(prod.device)
        mask_lable[torch.arange(prob_negs.size(0)), label] = False  
        prob_negs = prob_negs.masked_fill(~mask_lable, -1e6)

        # prob_negs[torch.arange(prod_pos.shape[0]), label]=-1e6  ## remove label
        
        prob_head = prob_negs[:, self.notail_items]
        ### TopK sample from head item, best
        indx_head_onehot, _ = self.gumbel_topk(prob_head, k)
        
        ### Random sample from head item
        # indx_head_onehot, _ = self.gumbel_randomk(prob_head, k)
        
        indx_head_onehot = indx_head_onehot.sum(dim=1)
        neg_pos_prob = torch.sum(indx_head_onehot*torch.exp(prob_head/tau), dim=-1) + prod_pos.squeeze(-1)
        loss_lpo = prod_pos.squeeze(-1)/neg_pos_prob
        
        ### Tail label loss
        # loss_lpo = tail_flag * loss_lpo
        # if torch.sum(tail_flag)>0:
        #     return torch.sum(loss_lpo)/torch.sum(tail_flag)
        # else:
        #     return 0
        
        # return loss_lpo.mean()
        return loss_lpo

  
    def training_step(self, batch, batch_idx):
        
        masks = torch.stack(batch['mask_seq'], dim=-1)
        seq = torch.stack(batch['seq_id'], dim=1)
        seq_rep = self.item_emb(seq)
        
        # _, rep = self.rec_model(seq_rep, masks)
        # prod = torch.matmul(rep, self.item_emb.weight.transpose(0, 1))
        
        prod, rep_seq = self.rec_model(seq_rep, masks)
        
        # loss_ce = self.loss_ce_raw(prod, batch['label_id'].long())
        loss_ce = self.loss_ce(prod, batch['label_id'].long()) ## none reduction
        # lpo_loss = self.lpo_loss(batch['userid'], prod, batch['label_id'].long(), batch['cold_item_flag'])
        lpo_loss = self.lpo_loss_gumbel(batch['userid'], prod, batch['label_id'].long(), batch['cold_item_flag'])
        
        # loss = loss_ce + 0.5*lpo_loss
        
        # loss = loss_ce
       
        ## plenty for sequences, best now
        ## loss = self.loss_ce(prod, batch['label_id'].long())
        if batch['cold_item_flag'].sum() == 0:
            tail_weight = 1.0
        else:
            tail_weight = torch.softmax(batch['plenty']/1, dim=0)
        
        # loss = (tail_weight*loss_ce).sum()
    
        # loss = (tail_weight*loss_ce).sum() + 0.01*lpo_loss
        
        loss = (tail_weight*(loss_ce + 0.5*lpo_loss)).sum() 
        
        
        self.log("train_loss", loss.item(),  on_epoch=False, prog_bar=True, logger=True)
        if self.trainer.global_rank == 0 and self.global_step == 100:
            report_memory('Seq rec')
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        self.rec_model.eval()
        masks = torch.stack(batch['mask_seq'], dim=-1)
        seq = torch.stack(batch['seq_id'], dim=1)
        seq_rep = self.item_emb(seq)
        prod, _ = self.rec_model(seq_rep, masks)
        
        # _, rep = self.rec_model(seq_rep, masks)
        # prod = torch.matmul(rep, self.item_emb.weight.transpose(0, 1))
        
        cold_idx = torch.where(batch['cold_item_flag']==1)
        metrics = calculate_metrics(prod, batch['label_id'].unsqueeze(1), metric_ks=[5, 10, 20, 50])
        return {"metrics": metrics, "prod_cold": prod[cold_idx], "label_cold": batch['label_id'][cold_idx]}
    
    def validation_epoch_end(self, validation_step_outputs):
        print('validation_epoch_end')
    
        prod_colds, label_colds = [], []
        for temp in validation_step_outputs:
            prod_colds.append(temp['prod_cold'])
            label_colds.append(temp['label_cold'])
        prod_colds = torch.cat(prod_colds, dim=0)
        label_colds = torch.cat(label_colds, dim=0)
        if len(label_colds) != 0:
            metrics_cold = calculate_metrics(prod_colds, label_colds.unsqueeze(1), metric_ks=[5, 10, 20, 50])
            print('tail item----------------------------------')
            for key_temp in metrics_cold:
                metrics_cold[key_temp] = round(np.mean(metrics_cold[key_temp] ) * 100, 4)
            print(metrics_cold)
            self.log("Tail Val_Metrics", metrics_cold)
            self.log("Tail HR@10", metrics_cold['HR@10'])
            self.logger_save.info("Tail Val Metrics: {}".format(metrics_cold))
            self.logger_save.info("Tail HR@10: {}".format(metrics_cold['HR@10']))
            
        # metrics_all = self.all_gather([i['metrics'] for i in validation_step_outputs])
        val_metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': [], 'HR@50': [], 'NDCG@50': []}
        val_metrics_dict_mean = {}
        for temp in validation_step_outputs:
            for key_temp, val_temp in temp['metrics'].items():
                val_metrics_dict[key_temp].append(val_temp)

        for key_temp, values_temp in val_metrics_dict.items():
            values_mean = round(np.mean(values_temp) * 100, 4)
            val_metrics_dict_mean[key_temp] = values_mean
        print('All -----------------------------------------')
        print(val_metrics_dict_mean)
        self.log("Val_Metrics", val_metrics_dict_mean)
        self.log("HR@10", val_metrics_dict_mean['HR@10'])
        self.logger_save.info("Val Metrics: {}".format(val_metrics_dict_mean))
        self.logger_save.info("HR@10: {}".format(val_metrics_dict_mean['HR@10']))
        
    def test_step(self, batch, batch_idx):
        self.rec_model.eval()
        masks = torch.stack(batch['mask_seq'], dim=-1)
        seq = torch.stack(batch['seq_id'], dim=1)
        seq_rep = self.item_emb(seq)
        prod, _ = self.rec_model(seq_rep, masks)
        self.prob_test.append(prod)
        # _, rep = self.rec_model(seq_rep, masks)
        # prod = torch.matmul(rep, self.item_emb.weight.transpose(0,1))
        metrics = calculate_metrics(prod, batch['label_id'].unsqueeze(1), metric_ks=[5, 10, 20, 50])
        cold_idx = torch.where(batch['cold_item_flag']==1)
        return {"metrics": metrics, "prod_cold": prod[cold_idx], "label_cold": batch['label_id'][cold_idx]}

    def test_epoch_end(self, test_step_outputs):
        print('test_epoch_end')
        prod_colds, label_colds = [], []
        for temp in test_step_outputs:
            prod_colds.append(temp['prod_cold'])
            label_colds.append(temp['label_cold'])
        prod_colds = torch.cat(prod_colds, dim=0)
        label_colds = torch.cat(label_colds, dim=0)
        metrics_cold = calculate_metrics(prod_colds, label_colds.unsqueeze(1), metric_ks=[5, 10, 20, 50])
        print('tail item----------------------------------')
        for key_temp in metrics_cold:
            metrics_cold[key_temp] = round(np.mean(metrics_cold[key_temp] ) * 100, 4)
        print(metrics_cold)
        self.log("Tail Test_Metrics", metrics_cold)
        # self.log("Tail HR@10", metrics_cold['HR@10'])
        self.logger_save.info("Tail Test Metrics: {}".format(metrics_cold))
        self.logger_save.info("Tail HR@10: {}".format(metrics_cold['HR@10']))
        
        test_metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': [], 'HR@50': [], 'NDCG@50': []}
        test_metrics_dict_mean = {}
        # metrics_all = self.all_gather([i['metrics'] for i in test_step_outputs])
        for temp in test_step_outputs:
            for key_temp, val_temp in temp['metrics'].items():
                test_metrics_dict[key_temp].append(val_temp)

        for key_temp, values_temp in test_metrics_dict.items():
            values_mean = round(np.mean(values_temp) * 100, 4)
            test_metrics_dict_mean[key_temp] = values_mean
        print('All -----------------------------------------')
        print(test_metrics_dict_mean)
        
        self.log("Test_Metrics", test_metrics_dict_mean)
        # self.log("HR@10", test_metrics_dict_mean['HR@10'])
        self.logger_save.info("Test Metrics: {}".format(test_metrics_dict_mean))
        self.logger_save.info("HR@10: {}".format(test_metrics_dict_mean['HR@10']))
        
        ### Toy example
        # plot_distribute(torch.cat(self.prob_test, dim=0), self.tail_items)
        quit()
        

def main():
    args_parser = argparse.ArgumentParser()
    args_parser = add_optimizer_args(args_parser)
    args_parser = add_datasets_args(args_parser)
    args_parser = UniversalDataModule.add_data_specific_args(args_parser)
    args_parser = Trainer.add_argparse_args(args_parser)
    args_parser = UniversalCheckpoint.add_argparse_args(args_parser)
    args_parser = TailImgRec.add_module_specific_args(args_parser)
    custom_parser = [
        '--datasets_path_train', '/data/Beauty/imgs_seq_5_train.pickle',
        '--datasets_path_test', '/data/Beauty/imgs_seq_5_test.pickle',
        '--datasets_path_val', '/data/Beauty/imgs_seq_5_val.pickle',
        '--datasets_name', 'Beauty',
        '--train_batchsize', '128',
        '--val_batchsize', '128',
        '--test_batchsize', '128',
        '--seq_len', '10',
        '--info', 'Tail 0.2, regterm tau 0.1',
        '--learning_rate', '5e-4',
        '--min_learning_rate', '5e-5',
        '--random_seed', '512',
        '--dropout', '0.8', 
        '--tau', '0.5',
        '--hidden_size', '768',
        '--attn_head', '16',
        '--n_blocks', '1',
        '--max_epochs', '100',
        '--save_ckpt_path', 'ckpt/temp'
        ] 
    
    args = args_parser.parse_args(args=custom_parser)
    fix_random_seed_as(args.random_seed)
    
    if not os.path.exists('../log/'):
        os.makedirs('../log/')
    if not os.path.exists('../log/' + args.datasets_name):
        os.makedirs('../log/' + args.datasets_name)
    logging.basicConfig(level=logging.INFO, filename='../log/' + args.datasets_name + '/' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '.log',
                    datefmt='%Y/%m/%d %H:%M:%S', format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s', filemode='w')
    logger = logging.getLogger(__name__)
    
    print(args.info)
    logger.info(args.info)
    print(args)
    logger.info(args)
    datasets = load_data(args)
    args.train_nums = len(datasets['train'])
    notail_items = notail_item(datasets['train'], 0.2)
    
    collate_fn = Collator(args, notail_items)
    args.item_nums = len(collate_fn.id_asin_dict)
    
    datamodule = UniversalDataModule(collate_fn=collate_fn, args=args, datasets=datasets)
    
    checkpoint_callback = UniversalCheckpoint(args)
    early_stop_callback_step = EarlyStopping(monitor='HR@10', min_delta=0.00, patience=3, verbose=False, mode='max')
    trainer = Trainer(devices=1, accelerator="gpu", strategy=DDPStrategy(find_unused_parameters=True), callbacks=[checkpoint_callback, early_stop_callback_step], max_epochs=args.max_epochs,  check_val_every_n_epoch=1)
    
    model = TailImgRec(args, collate_fn.tail_item, collate_fn.notail_item, datasets['train'].__len__(),logger)
    
    """
    ## Plot Distribution
    model_ckpt = torch.load('/ckpt/SASRec/last.ckpt')
  
    ### del ref. model
    # keys_to_delete = [k for k in model_ckpt['state_dict'] if 'ref_model' in k]
    # for k in keys_to_delete:
    #     del model_ckpt['state_dict'][k]
        
    model.load_state_dict(model_ckpt['state_dict'])
    trainer.test(model, datamodule)
    quit()
    """
    
    trainer.fit(model, datamodule)
    print(args)
   
    ## model_save
    # torch.save(model.state_dict(), args.datasets_name+'temp.pt')
    
    print('Test-------------------------------------------------------')
    trainer.test(model, datamodule)
    

if __name__ == '__main__':
    main()
