import argparse
import sys
import os
import csv
import time 
import torch
import networkx as nx
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from modules.data_loader import get_index_loader_test
from models import simpleGNN_MR
import modules.mod_utls as m_utls
from modules.loss import nll_loss, l2_regularization, nll_loss_raw, con_nll_loss_raw, un_nll_loss_raw
from modules.evaluation import eval_pred
from modules.aux_mod import fixed_augmentation
from sklearn.metrics import f1_score
from modules.conv_mod import CustomLinear
from modules.mr_conv_mod import build_mlp
import numpy as np
from scipy import sparse
from numpy import random
import igraph as ig
import math
import pandas as pd
from functools import partial
import dgl
import warnings
import yaml
warnings.filterwarnings("ignore")


class SoftAttentionDrop(nn.Module):
    def __init__(self, args):
        super(SoftAttentionDrop, self).__init__()
        dim = args['hidden-dim']
        
        self.temp = args['trainable-temp']
        self.p = args['trainable-drop-rate']
        if args['trainable-model'] == 'proj':
            self.mask_proj = CustomLinear(dim, dim)
        else:
            self.mask_proj = build_mlp(in_dim=dim, out_dim=dim, p=args['mlp-drop'], final_act=False)
        
        self.detach_y = args['trainable-detach-y']
        self.div_eps = args['trainable-div-eps']
        self.detach_mask = args['trainable-detach-mask']
        
    def forward(self, feature, in_eval=False):
        mask = self.mask_proj(feature)

        y = torch.zeros_like(mask)
        k = round(mask.shape[1] * self.p)

        for _ in range(k):
            if self.detach_y:
                w = torch.zeros_like(y)
                w[y>0.5] = 1
                w = (1. - w).detach()
            else:
                w = (1. - y)
                
            logw = torch.log(w + 1e-12)
            y1 = (mask + logw) / self.temp
            y1 = y1 - torch.amax(y1, dim=1, keepdim=True)
            
            if self.div_eps:
                y1 = torch.exp(y1) / (torch.sum(torch.exp(y1), dim=1, keepdim=True) + args['trainable-eps'])
            else:
                y1 = torch.exp(y1) / torch.sum(torch.exp(y1), dim=1, keepdim=True)
                
            y = y + y1 * w
            
        mask = 1. - y
        mask = mask / (1. - self.p)
        
        if in_eval and self.detach_mask:
            mask = mask.detach()
            
        return feature * mask


def create_model(args, e_ts):
    if args['model'] == 'backbone':
        tmp_model = simpleGNN_MR(in_feats=args['node-in-dim'], hidden_feats=args['hidden-dim'], out_feats=args['node-out-dim'], 
                                 num_layers=args['num-layers'], e_types=e_ts, input_drop=args['input-drop'], hidden_drop=args['hidden-drop'], 
                                 mlp_drop=args['mlp-drop'], mlp12_dim=args['mlp12-dim'], mlp3_dim=args['mlp3-dim'], bn_type=args['bn-type'])
    else:
        raise
    tmp_model.to(args['device'])
            
    return tmp_model
'''
def compute_class_balanced_weights(graph, beta=0.999999):
    labels = graph.ndata['label'].cpu().numpy()
    num_classes = len(np.unique(labels))

    class_counts = np.bincount(labels, minlength=num_classes)
    class_counts = np.maximum(class_counts, 1)

    effective_num = (1.0 - np.power(beta, class_counts)) / (1.0 - beta)
    weights = 1.0 / effective_num
    weights = weights / np.sum(weights) 

    return torch.tensor(weights, dtype=torch.float32).to(graph.device)
'''

def compute_page_rank(graph, alpha=0.85, personalization=None):
    # 如果是异构图，先转换为同质图
    if len(graph.ntypes) > 1 or len(graph.etypes) > 1:
        g_homogeneous = dgl.to_homogeneous(graph)
    else:
        g_homogeneous = graph.cpu()  # 确保图对象在CPU上

    # 创建邻接矩阵，确保边索引也在CPU上
    src, dst = g_homogeneous.edges()
    src = src.cpu().numpy()
    dst = dst.cpu().numpy()

    adj_matrix = sparse.csr_matrix(
        (np.ones(len(src)), (src, dst)),
        shape=(g_homogeneous.num_nodes(), g_homogeneous.num_nodes())
    )
    
    # 将稀疏矩阵转换为igraph图
    ig_graph = ig.Graph(directed=True)
    ig_graph.add_vertices(g_homogeneous.num_nodes())
    ig_graph.add_edges(zip(src, dst))

    # 计算PageRank
    pagerank_scores = ig_graph.pagerank(damping=alpha)
    
    return pagerank_scores

def compute_class_balanced_weights(graph, beta=0.999999, pagerank_alpha=0.9, alpha_balance=0.999999):
    labels = graph.ndata['label'].cpu().numpy()
    num_classes = len(np.unique(labels))

    class_counts = np.bincount(labels, minlength=num_classes)
    class_counts = np.maximum(class_counts, 1)

    # 计算有效样本数量
    effective_num = (1.0 - np.power(beta, class_counts)) / (1.0 - beta)
    weights_effective = 1.0 / effective_num

    # 计算每个节点的PageRank值
    pagerank_scores = compute_page_rank(graph, alpha=pagerank_alpha)
    
    # 对PageRank值归一化
    pagerank_scores_normalized = (pagerank_scores - np.min(pagerank_scores)) / (np.max(pagerank_scores) - np.min(pagerank_scores) + 1e-10)

    # 按类别计算归一化后的PageRank均值
    pagerank_per_class = [
        pagerank_scores_normalized[labels == i].mean() if sum(labels == i) > 0 else 0.0
        for i in range(num_classes)
    ]

    # 对类别PageRank均值再归一化
    pagerank_per_class_normalized = (pagerank_per_class - np.min(pagerank_per_class)) / (np.max(pagerank_per_class) - np.min(pagerank_per_class) + 1e-10)

    # 综合权重公式：α * (1 / E_n) + (1 - α) * PR
    weights = [
        alpha_balance * weight_eff + (1 - alpha_balance) * pagerank 
        for weight_eff, pagerank in zip(weights_effective, pagerank_per_class)
    ]

    # 归一化综合权重
    weights_normalized = np.array(weights) / np.sum(weights)

    return torch.tensor(weights_normalized, dtype=torch.float32).to(graph.device)

def UDA_train_epoch(epoch, model, loss_func, graph, label_loader, unlabel_loader, optimizer, augmentor, args, weights=None):
    model.train()
    num_iters = args['train-iterations']
    
    sampler, attn_drop, ad_optim = augmentor
    
    unlabel_loader_iter = iter(unlabel_loader)
    label_loader_iter = iter(label_loader)
    
    for idx in range(num_iters):
        try:
            label_idx = next(label_loader_iter)
        except StopIteration:
            label_loader_iter = iter(label_loader)
            label_idx = next(label_loader_iter)
        try:
            unlabel_idx = next(unlabel_loader_iter)
        except StopIteration:
            unlabel_loader_iter = iter(unlabel_loader)
            unlabel_idx = next(unlabel_loader_iter)

        if epoch > args['trainable-warm-up']:
            model.eval()
            with torch.no_grad():
                _, _, u_blocks = fixed_augmentation(graph, unlabel_idx.to(args['device']), sampler, aug_type='none')
                weak_inter_results = model(u_blocks, update_bn=False, return_logits=True)
                weak_h = torch.stack(weak_inter_results, dim=1)
                weak_h = weak_h.reshape(weak_h.shape[0], -1)
                weak_logits = model.proj_out(weak_h)
                u_pred_weak_log = weak_logits.log_softmax(dim=-1)
                u_pred_weak = u_pred_weak_log.exp()[:, 1]
                
            pseudo_labels = torch.ones_like(u_pred_weak).long()
            neg_tar = (u_pred_weak <= (args['normal-th']/100.)).bool()
            pos_tar = (u_pred_weak >= (args['fraud-th']/100.)).bool()
            pseudo_labels[neg_tar] = 0
            pseudo_labels[pos_tar] = 1
            u_mask = torch.logical_or(neg_tar, pos_tar)

            model.train()
            attn_drop.train()
            for param in model.parameters():
                param.requires_grad = False
            for param in attn_drop.parameters():
                param.requires_grad = True

            _, _, u_blocks = fixed_augmentation(graph, unlabel_idx.to(args['device']), sampler, aug_type='drophidden')

            inter_results = model(u_blocks, update_bn=False, return_logits=True)
            dropped_results = [inter_results[0]]
            for i in range(1, len(inter_results)):
                dropped_results.append(attn_drop(inter_results[i]))
            h = torch.stack(dropped_results, dim=1)
            h = h.reshape(h.shape[0], -1)
            logits = model.proj_out(h)
            u_pred = logits.log_softmax(dim=-1)
            
            # 使用类平衡权重计算一致性损失
            consistency_loss = con_nll_loss_raw(u_pred, pseudo_labels, pos_w=1.0, reduction='none')
            print(f"Class weights (consistency loss): {weights}")  # 打印类平衡权重
            consistency_loss = torch.mean(consistency_loss * u_mask)

            if args['diversity-type'] == 'cos':
                diversity_loss = F.cosine_similarity(weak_h, h, dim=-1)
            elif args['diversity-type'] == 'euc':
                diversity_loss = F.pairwise_distance(weak_h, h)
            else:
                raise ValueError("Unsupported diversity type")

            diversity_loss = torch.mean(diversity_loss * u_mask)
            
            total_loss = args['trainable-consis-weight'] * consistency_loss - diversity_loss + args['trainable-weight-decay'] * l2_regularization(attn_drop)
            
            ad_optim.zero_grad()
            total_loss.backward()
            ad_optim.step()
            
            for param in model.parameters():
                param.requires_grad = True
            for param in attn_drop.parameters():
                param.requires_grad = False

            inter_results = model(u_blocks, update_bn=False, return_logits=True)
            dropped_results = [inter_results[0]]
            for i in range(1, len(inter_results)):
                dropped_results.append(attn_drop(inter_results[i], in_eval=True))

            h = torch.stack(dropped_results, dim=1)
            h = h.reshape(h.shape[0], -1)
            logits = model.proj_out(h)
            u_pred = logits.log_softmax(dim=-1)

            # 使用类平衡权重计算无监督损失
            unsup_loss = F.cross_entropy(logits, pseudo_labels, weight=weights, reduction='none')
            print(f"Class weights (unsupervised loss): {weights}")  # 打印类平衡权重
            unsup_loss = torch.mean(unsup_loss * u_mask)
        else:
            unsup_loss = 0.0

        _, _, s_blocks = fixed_augmentation(graph, label_idx.to(args['device']), sampler, aug_type='none')
        s_pred = model(s_blocks)
        s_target = s_blocks[-1].dstdata['label']
            
        # 使用类平衡权重计算有监督损失
        sup_loss, _ = loss_func(s_pred, s_target)
        print(f"Class weights (supervised loss): {weights}")  # 打印类平衡权重

        loss = sup_loss + unsup_loss + args['weight-decay'] * l2_regularization(model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()     

    print(f"Epoch {epoch + 1}, Loss: {sup_loss.item():.4f}")        

def get_model_pred(model, graph, data_loader, sampler, args):
    model.eval()
    
    pred_list = []
    target_list = []
    with torch.no_grad():
        for node_idx in data_loader:
            _, _, blocks = sampler.sample_blocks(graph, node_idx.to(args['device']))
            
            pred = model(blocks)
            target = blocks[-1].dstdata['label']
            
            pred_list.append(pred.detach())
            target_list.append(target.detach())
        pred_list = torch.cat(pred_list, dim=0)
        target_list = torch.cat(target_list, dim=0)
        pred_list = pred_list.exp()[:, 1]
        
    return pred_list, target_list


def val_epoch(epoch, model, graph, valid_loader, test_loader, sampler, args):
    valid_dict = {}
    valid_pred, valid_target = get_model_pred(model, graph, valid_loader, sampler, args)
    v_roc, v_pr, _, _, _, v_f1, v_thre = eval_pred(valid_pred, valid_target)
    valid_dict['auc-roc'] = v_roc
    valid_dict['auc-pr'] = v_pr
    valid_dict['marco f1'] = v_f1
        
    test_dict = {}
    test_pred, test_target = get_model_pred(model, graph, test_loader, sampler, args)
    t_roc, t_pr, _, _, _, _, _ = eval_pred(test_pred, test_target)
    test_dict['auc-roc'] = t_roc
    test_dict['auc-pr'] = t_pr
    
    test_pred = test_pred.cpu().numpy()
    test_target = test_target.cpu().numpy()
    guessed_target = np.zeros_like(test_target)
    guessed_target[test_pred > v_thre] = 1
    t_f1 = f1_score(test_target, guessed_target, average='macro')
    test_dict['marco f1'] = t_f1
            
    return valid_dict, test_dict

# 修改后的主运行函数
def run_model(args):
    graph, label_loader, valid_loader, test_loader, unlabel_loader = get_index_loader_test(
        name=args['data-set'], 
        batch_size=args['batch-size'], 
        unlabel_ratio=args['unlabel-ratio'],
        training_ratio=args['training-ratio'],
        shuffle_train=args['shuffle-train'], 
        to_homo=args['to-homo']
    )
    graph = graph.to(args['device'])
    
    args['node-in-dim'] = graph.ndata['feature'].shape[1]
    args['node-out-dim'] = 2
    
    my_model = create_model(args, graph.etypes)
    
    if args['optim'] == 'adam':
        optimizer = torch.optim.Adam(my_model.parameters(), lr=args['lr'], weight_decay=0.0)
    elif args['optim'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(my_model.parameters(), lr=args['lr'], weight_decay=0.0)
    
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args['num-layers'])
    
    train_epoch = UDA_train_epoch
    attn_drop = SoftAttentionDrop(args).to(args['device'])
    if args['trainable-optim'] == 'rmsprop':
        ad_optim = torch.optim.RMSprop(attn_drop.parameters(), lr=args['trainable-lr'], weight_decay=0.0)
    else:
        ad_optim = torch.optim.Adam(attn_drop.parameters(), lr=args['trainable-lr'], weight_decay=0.0)
    augmentor = (sampler, attn_drop, ad_optim)

    task_loss = nll_loss
    
    best_val = sys.float_info.min

    # 在训练开始前计算类别权重
    weights = compute_class_balanced_weights(graph, beta=args.get('beta', 0.999999))

    for epoch in range(args['epochs']):
        train_epoch(epoch, my_model, task_loss, graph, label_loader, unlabel_loader, optimizer, augmentor, args, weights=weights)
        val_results, test_results = val_epoch(epoch, my_model, graph, valid_loader, test_loader, sampler, args)
        
        if val_results['auc-roc'] > best_val:
            best_val = val_results['auc-roc']
            test_in_best_val = test_results
            
            if args['store-model']:
                m_utls.store_model(my_model, args)
                
    return list(test_in_best_val.values())

def get_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.load(setting, Loader=yaml.FullLoader)
    return config


if __name__ == '__main__':
    start_time = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str, help='Path to the config file.')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs. Default is 1.')
    cfg = vars(parser.parse_args())
    
    args = get_config(cfg['config'])
    if torch.cuda.is_available():
        args['device'] = torch.device('cuda:%d'%(args['device']))
    else:
        args['device'] = torch.device('cpu')
                                            
    print(args)
    final_results = []
    for r in range(cfg['runs']):
        final_results.append(run_model(args))
        
    final_results = np.array(final_results)
    mean_results = np.mean(final_results, axis=0)
    std_results = np.std(final_results, axis=0)

    print(mean_results)
    print(std_results)
    print('total time: ', time.time()-start_time)
