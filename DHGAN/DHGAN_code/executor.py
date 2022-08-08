import os
import numpy as np
import torch
import math
import dgl
import datetime
from eval import *
from layers.decgcn import DecGCN
from layers.graphsage import Graphsage
from layers.hyperbolic import HyperbolicGCN
from layers.loss import DECGCNLoss, HyperbolicLoss, HATLoss
from optimizer.radam import RiemannianAdam
from layers.gat import GAT
from layers.hgnn import HGNN
from layers.han import HAN
from layers.hat import HAT
from layers.gatne import GATNE
from layers.graphsage_simcor import GraphsageSimCor
from hyperbolic.PoincareManifold import PoincareManifold

class Executor(object):
    def __init__(self, args):
        self.args = args
        self.poincare = PoincareManifold(0, 0)
        self.device = self._acquire_device()
        self.graph_data = self._get_graph_data()  ## get graph data，item-cor-item item-sim-item
        #self.grapg_data = self.remove_relation()  ## remove relation
        #self.graph_data = self.reverse_relation()  ## reverse relation

        if args.run_mode != 'eval':
            self.nfeat = self.graph_data.ndata.pop('sparse_feats')
            if args.sparse_maxs == 'auto':
                sparse_maxs = []
                for i in range(self.nfeat.shape[1]):
                    sparse_maxs.append(torch.max(self.nfeat[:, i]).item() + 1)
                args.sparse_maxs = sparse_maxs ## [3,98,39958]
            else:
                args.sparse_maxs = [int(x) for x in args.sparse_maxs.split(',')]
            if args.sparse_emb_dim == 'auto':
                sparse_emb_dim = []
                for m in sparse_maxs:
                    sparse_emb_dim.append(math.ceil(math.log2(m)))
                args.sparse_emb_dim = sparse_emb_dim  ## [2,7,16]
            else:
                args.sparse_emb_dim = [int(x) for x in args.sparse_emb_dim.split(',')]
            self.model = self._build_model().to(self.device)  ##  model

    def _build_model(self):
        if self.args.model == 'graphsage':
            model = Graphsage(node_nums=self.graph_data.num_nodes(),
                            nodeid_emb_dim=self.args.nodeid_emb_dim,
                            sparse_maxs=self.args.sparse_maxs,
                            sparse_emb_dim=self.args.sparse_emb_dim,
                            hidden_dim=self.args.hidden_dim,
                            output_dim=self.args.output_dim,
                            n_layers=self.args.num_layers,
                            negs_nums=self.args.negs_nums,
                            dropout=self.args.dropout,
                            etypes=self.graph_data.etypes,
                            use_nodeid_emb=self.args.use_nodeid_emb)
        elif self.args.model == 'gat':
            model = GAT(node_nums=self.graph_data.num_nodes(),
                            nodeid_emb_dim=self.args.nodeid_emb_dim,
                            sparse_maxs=self.args.sparse_maxs,
                            sparse_emb_dim=self.args.sparse_emb_dim,
                            hidden_dim=self.args.hidden_dim,
                            output_dim=self.args.output_dim,
                            n_layers=self.args.num_layers,
                            negs_nums=self.args.negs_nums,
                            dropout=self.args.dropout,
                            etypes=self.graph_data.etypes,
                            use_nodeid_emb=self.args.use_nodeid_emb)
        elif self.args.model == 'han':
            model = HAN(node_nums=self.graph_data.num_nodes(),
                            nodeid_emb_dim=self.args.nodeid_emb_dim,
                            sparse_maxs=self.args.sparse_maxs,
                            sparse_emb_dim=self.args.sparse_emb_dim,
                            hidden_dim=self.args.hidden_dim,
                            output_dim=self.args.output_dim,
                            n_layers=self.args.num_layers,
                            negs_nums=self.args.negs_nums,
                            dropout=self.args.dropout,
                            etypes=self.graph_data.etypes,
                            use_nodeid_emb=self.args.use_nodeid_emb)
        elif self.args.model == 'gatne':
            model = GATNE(node_nums=self.graph_data.num_nodes(),
                            nodeid_emb_dim=self.args.nodeid_emb_dim,
                            sparse_maxs=self.args.sparse_maxs,
                            sparse_emb_dim=self.args.sparse_emb_dim,
                            hidden_dim=self.args.hidden_dim,
                            output_dim=self.args.output_dim,
                            n_layers=self.args.num_layers,
                            negs_nums=self.args.negs_nums,
                            dropout=self.args.dropout,
                            etypes=self.graph_data.etypes,
                            use_nodeid_emb=self.args.use_nodeid_emb)
        elif self.args.model == 'graphsage_simcor':
            model = GraphsageSimCor(node_nums=self.graph_data.num_nodes(),
                           nodeid_emb_dim=self.args.nodeid_emb_dim,
                           sparse_maxs=self.args.sparse_maxs,
                           sparse_emb_dim=self.args.sparse_emb_dim,
                           hidden_dim=self.args.hidden_dim,
                           output_dim=self.args.output_dim,
                           n_layers=self.args.num_layers,
                           negs_nums=self.args.negs_nums,
                           dropout=self.args.dropout,
                           etypes=self.graph_data.etypes,
                           use_nodeid_emb=self.args.use_nodeid_emb)
        elif self.args.model == 'decgcn':
            model = DecGCN(node_nums=self.graph_data.num_nodes(),
                           nodeid_emb_dim=self.args.nodeid_emb_dim,
                           sparse_maxs=self.args.sparse_maxs,
                           sparse_emb_dim=self.args.sparse_emb_dim,
                           hidden_dim=self.args.hidden_dim,
                           output_dim=self.args.output_dim,
                           n_layers=self.args.num_layers,
                           negs_nums=self.args.negs_nums,
                           dropout=self.args.dropout,
                           etypes=self.graph_data.etypes,
                           fanout=self.args.fanout,
                           use_nodeid_emb=self.args.use_nodeid_emb)
        elif self.args.model == 'hyperbolic':
            model = HyperbolicGCN(node_nums=self.graph_data.num_nodes(),
                           nodeid_emb_dim=self.args.nodeid_emb_dim,
                           sparse_maxs=self.args.sparse_maxs,
                           sparse_emb_dim=self.args.sparse_emb_dim,
                           hidden_dim=self.args.hidden_dim,
                           output_dim=self.args.output_dim,
                           n_layers=self.args.num_layers,
                           negs_nums=self.args.negs_nums,
                           dropout=self.args.dropout,
                           etypes=self.graph_data.etypes,
                           fanout=self.args.fanout,
                           use_nodeid_emb=self.args.use_nodeid_emb)
        elif self.args.model == 'hgnn':
            model = HGNN(node_nums=self.graph_data.num_nodes(),
                           nodeid_emb_dim=self.args.nodeid_emb_dim,
                           sparse_maxs=self.args.sparse_maxs,
                           sparse_emb_dim=self.args.sparse_emb_dim,
                           hidden_dim=self.args.hidden_dim,
                           output_dim=self.args.output_dim,
                           n_layers=self.args.num_layers,
                           negs_nums=self.args.negs_nums,
                           dropout=self.args.dropout,
                           etypes=self.graph_data.etypes,
                           use_nodeid_emb=self.args.use_nodeid_emb)
        elif self.args.model == 'hat':
            model = HAT(node_nums=self.graph_data.num_nodes(),
                           nodeid_emb_dim=self.args.nodeid_emb_dim,
                           sparse_maxs=self.args.sparse_maxs,
                           sparse_emb_dim=self.args.sparse_emb_dim,
                           hidden_dim=self.args.hidden_dim,
                           output_dim=self.args.output_dim,
                           n_layers=self.args.num_layers,
                           negs_nums=self.args.negs_nums,
                           dropout=self.args.dropout,
                           etypes=self.graph_data.etypes,
                           use_nodeid_emb=self.args.use_nodeid_emb)
        else:
            assert False, 'not support `{}`'.format(self.args.model)
        return model

    def _acquire_device(self):
        if self.args.device == 'cuda':
            device_num = '1'
            os.environ["CUDA_VISIBLE_DEVICES"] = device_num
            print('Use GPU: cuda:{}'.format(device_num))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_graph_data(self):
        g = dgl.load_graphs(os.path.join(self.args.graph_root_path, self.args.graph_data_file))[0][0]
        # print('Total node num: {}\nTotal edge num: {}; sim: {}; cor: {}\n'.
        #       format(g.num_nodes('item'), g.num_edges('sim') + g.num_edges('cor'), g.num_edges('sim'),
        #              g.num_edges('cor')))
        return g
    
    
    def reverse_relation(self):    
        for etype in self.graph_data.etypes:
            train_mask = self.graph_data.edges[etype].data['train_mask']
            mask_np = torch.nonzero(train_mask).numpy()
            np.random.shuffle(mask_np)
            mask_np = torch.from_numpy(mask_np)
            mask_np = mask_np.squeeze()
            edges_src, edges_dst, edges = self.graph_data.edges(form='all', etype=etype)
            self.graph_data = dgl.remove_edges(self.graph_data, mask_np[:int(self.args.remove_ratio*len(mask_np))], etype = etype)
            tmp_type = 'sim' if etype == 'cor' else 'cor'
            self.graph_data.add_edges(edges_src[mask_np[:int(self.args.remove_ratio*len(mask_np))]], edges_dst[mask_np[:int(self.args.remove_ratio*len(mask_np))]], etype = tmp_type )
            train_mask = self.graph_data.edges[etype].data['train_mask']
        return self.graph_data
    
    def remove_relation(self):
        for etype in self.graph_data.etypes:
            
            train_mask = self.graph_data.edges[etype].data['train_mask']
            mask_np = torch.nonzero(train_mask).numpy()
            np.random.shuffle(mask_np)
            mask_np = torch.from_numpy(mask_np)
            mask_np = mask_np.squeeze()
            self.graph_data = dgl.remove_edges(self.graph_data, mask_np[:int(self.args.remove_ratio*len(mask_np))], etype = etype)
            train_mask = self.graph_data.edges[etype].data['train_mask']
            print(len(train_mask))
        return self.graph_data
    


    def _get_dataloader_train(self):
        train_seeds = {}  ##
        for etype in self.graph_data.etypes:
            train_mask = self.graph_data.edges[etype].data['train_mask']
            train_seeds[etype] = torch.LongTensor(torch.nonzero(train_mask)).squeeze() # torch.Size([2542888])
        # train_seeds1 = {e: np.arange(self.graph_data.num_edges(e)) for e in edges_type}
        fanouts = [{e: fanout for e in self.graph_data.etypes} for fanout in self.args.fanout] ## sample fanout neighbors for each type of edge for each level
        sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)
        dataloader = dgl.dataloading.EdgeDataLoader(
            self.graph_data, train_seeds, sampler,
            negative_sampler=dgl.dataloading.negative_sampler.Uniform(self.args.negs_nums),
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=self.args.num_workers)
        return dataloader

    def _get_dataloader_saveemb(self):
        nids = np.arange(self.graph_data.num_nodes())
        fanouts = [{e: fanout for e in self.graph_data.etypes} for fanout in self.args.fanout]
        sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)
        dataloader = dgl.dataloading.NodeDataLoader(self.graph_data, nids, sampler,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=self.args.num_workers)
        return dataloader

    def _select_optimizer(self):
        model_optim = RiemannianAdam(params=self.model.parameters(), lr=self.args.learning_rate)
        # model_optim = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        # model_optim = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate)
        if self.args.model == 'gatne':
            model_optim = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        if self.args.model == 'decgcn':
            model_optim = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        if self.args.model == 'gat':
            model_optim = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.model == 'hyperbolic':
            criterion = HyperbolicLoss(neg_nums=self.args.negs_nums, edge_types=self.graph_data.etypes)
        elif self.args.model == 'hat':
            criterion = HATLoss(neg_nums=self.args.negs_nums, edge_types=self.graph_data.etypes)
        elif self.args.model == 'gat':
            criterion = DECGCNLoss(neg_nums=self.args.negs_nums, edge_types=self.graph_data.etypes)
        elif self.args.model == 'hgnn':
            criterion = DECGCNLoss(neg_nums=self.args.negs_nums, edge_types=self.graph_data.etypes)
        elif self.args.model == 'decgcn':
            criterion = DECGCNLoss(neg_nums=self.args.negs_nums, edge_types=self.graph_data.etypes)
        elif self.args.model == 'gatne':
            criterion = DECGCNLoss(neg_nums=self.args.negs_nums, edge_types=self.graph_data.etypes)
        else:
            criterion = HyperbolicLoss(neg_nums=self.args.negs_nums, edge_types=self.graph_data.etypes)
        return criterion

    def train(self):
        begin_info = '\n   Begin Time={} | Model={} Dataset={}\n   sparse_emb_dim={} hidden_dim={} output_dim={}'.\
            format(datetime.datetime.now(), self.args.model, self.args.dataset,
                   self.args.nodeid_emb_dim if self.args.use_nodeid_emb else -1,
                   str(self.args.sparse_emb_dim), self.args.hidden_dim, self.args.output_dim)
        self.logs('train.log', begin_info)  ## logging
        self.model.train()  ## train mode
        #print(self.model)
        dataloader = self._get_dataloader_train()  
        criterion = self._select_criterion()
        optimizer = self._select_optimizer()
        for epoch in range(self.args.train_epochs):
            for idx, (input_nodes, pos_graph, neg_graph, blocks) in enumerate(dataloader):
                
                batch_inputs = self.nfeat[input_nodes].to(self.args.device)
                if self.args.use_nodeid_emb:
                    batch_inputs = torch.cat([batch_inputs, input_nodes.to(self.args.device).unsqueeze(dim=1)], dim=1)
                pos_graph = pos_graph.to(self.args.device)
                neg_graph = neg_graph.to(self.args.device)
                blocks = [block.int().to(self.args.device) for block in blocks]
                batch_pred = self.model(blocks, batch_inputs)
                loss, mrr = criterion(batch_pred, pos_graph, neg_graph)
                if (idx+1) % 100 == 0:
                    self.epoch_hook(epoch_num=epoch, batch_num=idx)
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                self.batch_hook(epoch_num=epoch, batch_num=idx, loss=loss.item(), mrr=mrr.item())
            #self.epoch_hook(epoch_num=epoch, batch_num=0)

    def save_embedding(self):
        test_model_file = torch.load(os.path.join(self.args.checkpoints, self.args.dataset, '{}_{}.pkl'.format(self.args.model_name, self.args.embedding_use_model)))
        self.model.load_state_dict(test_model_file)
        self.model = self.model.to(self.args.device)

        dataloader = self._get_dataloader_saveemb()
        nodes_emb = {k: torch.zeros([self.graph_data.num_nodes(), self.args.output_dim], device=self.args.device) for k in self.graph_data.etypes}

        self.model.eval()
        with torch.no_grad():
            for idx, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                batch_inputs = self.nfeat[input_nodes].to(self.args.device)
                if self.args.use_nodeid_emb:
                    batch_inputs = torch.cat([batch_inputs, input_nodes.to(self.args.device).unsqueeze(dim=1)], dim=1)
                blocks = [block.int().to(self.args.device) for block in blocks]
                batch_pred = self.model(blocks, batch_inputs)
                # batch_nid = input_nodes[:blocks[1].number_of_dst_nodes()]
                batch_nid = output_nodes
                for etype in batch_pred.keys():
                    nodes_emb[etype][batch_nid] = batch_pred[etype]
                print('{}/{}'.format(idx + 1, len(dataloader)))

        emb_savepath = os.path.join(self.args.embedding_save_path, self.args.dataset)
        if not os.path.exists(emb_savepath):
            os.makedirs(emb_savepath)
        for etype in self.graph_data.etypes:
            np_array = nodes_emb[etype].cpu().detach().numpy()
            np.save(os.path.join(emb_savepath, '{}_{}.npy'.format(self.args.model_name, etype)), np_array)

    def eval(self):
        begin_info = '\n   Begin Time={} | Model={} Dataset={}\n   embedding_use_model={} eval_neg_nums={} eval_topk={}'. \
            format(datetime.datetime.now(), self.args.model, self.args.dataset,
                   self.args.embedding_use_model, self.args.eval_neg_nums, self.args.eval_topk)
        self.logs('eval.log', begin_info)
        k = self.args.eval_topk
        negs_nums = self.args.eval_neg_nums
        hr_total = {}
        mrr_total = {}
        ndcg_total = {}
        for etype in self.graph_data.etypes:
#             if etype == "cor":
#                 continue
            hr, mrr, ndcg = [], [], []
            g_test = dgl.edge_subgraph(self.graph_data,{etype: torch.tensor(self.graph_data.edges[etype].data['test_mask'], dtype=torch.bool)}, preserve_nodes=True)
            emb = np.load(os.path.join('./save_embedding', self.args.dataset, '{}_{}.npy'.format(self.args.model_name, etype)))

            #print(emb.shape)  ## [num_nodes * 32]
            #test_nids = g_test.ndata['_ID'].cpu().numpy()
            edges_src, edges_dst = g_test.edges(etype=etype)
            for idx in range(len(edges_src)):
                negs = negative_sampler(self.graph_data, nid=edges_src[idx], nid1 = edges_dst[idx], etype=etype, negs_nums=negs_nums) ##[10000 * 1]
                sample_nids = torch.cat([torch.tensor([edges_dst[idx]]), negs]) ##[10001 * 1]
                query_embs = emb[sample_nids] ##[10001 * 32]
                emb_S = torch.from_numpy(emb[edges_src[idx]].reshape((1,len(emb[edges_dst[idx]]))))  ## [1 * 32]
                query_embs = torch.tensor(query_embs) ## [1 * 32]
                # hyperbolic distance
                
                dis = self.poincare.distance(emb_S,query_embs) ## [10001 * 1]
               
                dis_n = dis.numpy()
            
                dis_V = dis_n.reshape(negs_nums + 1,)  ## [10001]
                ranks = np.argsort(dis_V)  ## [10001]
                #print(np.argwhere(ranks == 0))
                _hr, _mrr, _ndcg = scores(ranks[:k], target=0)
                hr.append(_hr)
                mrr.append(_mrr)
                ndcg.append(_ndcg)
                if (idx + 1) % 100 == 0:
                    print('etype:{}, edge:{}/{}'.format(etype, idx + 1, len(edges_src)))
            hr_total[etype] = sum(hr) / len(hr)
            mrr_total[etype] = sum(mrr) / len(mrr)
            ndcg_total[etype] = sum(ndcg) / len(ndcg)
            print(hr_total[etype])
            print(mrr_total[etype])
            print(ndcg_total[etype])
        for etype in self.graph_data.etypes:
            info = 'Model={}, Dataset={}\n{}: HitRate@{}: {} | MRR@{}: {} | NDCG@{}: {}'.\
                format(self.args.model, self.args.dataset, etype, k, hr_total[etype], k, mrr_total[etype], k, ndcg_total[etype])
            self.logs('eval.log', info)
            print(info)

    def batch_hook(self, epoch_num, batch_num, **kwargs):
        info = 'Model={}, Dataset={} | epoch={}, batch={} | loss={}, mrr={}'.\
            format(self.args.model, self.args.dataset, epoch_num, batch_num, kwargs['loss'], kwargs['mrr'])
        self.logs('train.log', info)
        print(info)

    def epoch_hook(self, epoch_num,  batch_num):
        save_path = os.path.join(self.args.checkpoints, self.args.dataset)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(self.model.state_dict(), os.path.join(save_path, '{}_{}_{}.pkl'.format(self.args.model_name, epoch_num, batch_num)))

    def logs(self, log_file, content):
        if not os.path.exists(self.args.log_path):
            os.makedirs(self.args.log_path)
        with open(os.path.join(self.args.log_path, log_file), mode='a') as f:
            f.write(content + '\t\n')

