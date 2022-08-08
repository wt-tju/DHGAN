import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


def mrr(logits, negs_logits):
    mrr_all = torch.cat((negs_logits, logits), dim=-1)
    mrr_size = mrr_all.shape[-1]
    _, indices_of_ranks = mrr_all.topk(mrr_size)
    _, ranks = (-indices_of_ranks).topk(mrr_size)
    _mrr = (ranks[:, -1] + 1).float().reciprocal().mean()
    return _mrr


class DECGCNLoss(nn.Module):
    def __init__(self, neg_nums, edge_types):
        super(DECGCNLoss, self).__init__()
        self.neg_nums = neg_nums
        self.edge_types = edge_types

    def forward(self, block_outputs, pos_graph, neg_graph):
        loss = []
        _mrr = []
        for etype in self.edge_types:
            with pos_graph.local_scope():
                pos_graph.ndata['h'] = block_outputs[etype]
                pos_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype) 
                pos_score = pos_graph.edata['score']
            with neg_graph.local_scope():
                neg_graph.ndata['h'] = block_outputs[etype]
                neg_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
                neg_score = neg_graph.edata['score']

            if pos_score[list(pos_score.keys())[0]].shape[0] == 0:
                continue
            pos_logits = pos_score[list(pos_score.keys())[0]]
            neg_logits = neg_score[list(neg_score.keys())[0]]
            #print(pos_logits)
            #print(neg_logits)
            _mrr.append(mrr(pos_logits, neg_logits.view(-1, self.neg_nums)))
            pos_loss = F.logsigmoid(pos_logits).squeeze(dim=1)
            negs_loss = F.logsigmoid(-neg_logits).squeeze(dim=1)
            loss.append(-(pos_loss.mean() + negs_loss.mean()))
        return sum(loss) / len(loss), sum(_mrr) / len(_mrr)


class HyperbolicLoss(nn.Module):
    def __init__(self, neg_nums, edge_types):
        super(HyperbolicLoss, self).__init__()
        self.neg_nums = neg_nums
        self.edge_types = edge_types

    def forward(self, block_outputs, pos_graph, neg_graph):
        loss = []
        _mrr = []
        for etype in self.edge_types:
            if etype in ['cor', 'sim']:
                with pos_graph.local_scope():
                    pos_graph.ndata['h'] = block_outputs[etype]
                    pos_graph.apply_edges(hyperbolic_dis, etype=etype)
                    pos_score = pos_graph.edata['score']
                with neg_graph.local_scope():
                    neg_graph.ndata['h'] = block_outputs[etype]
                    neg_graph.apply_edges(hyperbolic_dis, etype=etype)
                    neg_score = neg_graph.edata['score']
                if pos_score[list(pos_score.keys())[0]].shape[0] == 0:
                    continue
                pos_logits = pos_score[list(pos_score.keys())[0]]
                neg_logits = neg_score[list(neg_score.keys())[0]]
                _mrr.append(mrr(-pos_logits, -neg_logits.view(-1, self.neg_nums)))
                pos_loss = pos_logits.squeeze(dim=1)
                neg_loss = neg_logits.squeeze(dim=1)
                pos_loss = torch.log(1-torch.sigmoid(pos_loss)+1e-5)
                neg_loss = torch.log(torch.sigmoid(neg_loss)+1e-5)
                #pos_loss = -torch.log(pos_loss)
                #pos_loss = torch.log(neg_loss)
                loss_mean = -pos_loss.mean() - neg_loss.mean()
                loss.append(loss_mean)
                # loss.append(-(pos_loss.mean() + negs_loss.mean()))
            elif etype == 'corr':
                with pos_graph.local_scope():
                    pos_graph.ndata['h'] = block_outputs[etype]
                    pos_graph.apply_edges(hyperbolic_cosine, etype=etype)
                    pos_score = pos_graph.edata['score']
                with neg_graph.local_scope():
                    neg_graph.ndata['h'] = block_outputs[etype]
                    neg_graph.apply_edges(hyperbolic_cosine, etype=etype)
                    neg_score = neg_graph.edata['score']
                if pos_score[list(pos_score.keys())[0]].shape[0] == 0:
                    continue
                pos_logits = pos_score[list(pos_score.keys())[0]]
                neg_logits = neg_score[list(neg_score.keys())[0]]
                _mrr.append(mrr(pos_logits, neg_logits.view(-1, self.neg_nums)))
                pos_loss = pos_logits.squeeze(dim=1)
                neg_loss = neg_logits.squeeze(dim=1)

                loss_mean = -pos_loss.mean() + neg_loss.mean()
                loss.append(loss_mean)

        return sum(loss) / len(loss), sum(_mrr) / len(_mrr)

class HATLoss(nn.Module):
    def __init__(self, neg_nums, edge_types):
        super(HATLoss, self).__init__()
        self.neg_nums = neg_nums
        self.edge_types = edge_types

    def forward(self, block_outputs, pos_graph, neg_graph):
        loss = []
        _mrr = []
        for etype in self.edge_types:
            with pos_graph.local_scope():
                pos_graph.ndata['h'] = block_outputs[etype]
                pos_graph.apply_edges(hyperbolic_dis, etype=etype)
                pos_score = pos_graph.edata['score']
            with neg_graph.local_scope():
                neg_graph.ndata['h'] = block_outputs[etype]
                neg_graph.apply_edges(hyperbolic_dis, etype=etype)
                neg_score = neg_graph.edata['score']
            if pos_score[list(pos_score.keys())[0]].shape[0] == 0:
                continue
            pos_logits = pos_score[list(pos_score.keys())[0]]
            neg_logits = neg_score[list(neg_score.keys())[0]]
            _mrr.append(mrr(-pos_logits, -neg_logits.view(-1, self.neg_nums)))

            pos_loss = pos_logits.squeeze(dim=1)
            negs_loss = neg_logits.squeeze(dim=1)

            pos_loss = torch.log(1-torch.sigmoid(pos_logits))
            neg_loss = torch.log(torch.sigmoid(neg_logits))
            loss_mean = - pos_loss.mean() - neg_loss.mean()
            loss.append(loss_mean)
            # loss.append(-(pos_loss.mean() + negs_loss.mean()))
        return sum(loss) / len(loss), sum(_mrr) / len(_mrr)

def hyperbolic_dis(edges):
    # scores = torch.matmul(edges.src['h'].unsqueeze(dim=-2), edges.dst['h'].unsqueeze(dim=-1))
    from hyperbolic.PoincareManifold import PoincareManifold
    poincare = PoincareManifold(0, 0)
    src_emb, dis_emb = edges.src.pop('h'), edges.dst.pop('h')
    scores = poincare.distance(src_emb, dis_emb) ** 2
    return {'score': scores}

def hyperbolic_inner(edges):
    # scores = torch.matmul(edges.src['h'].unsqueeze(dim=-2), edges.dst['h'].unsqueeze(dim=-1))
    from hyperbolic.PoincareManifold import PoincareManifold
    poincare = PoincareManifold(0, 0)
    src_emb, dis_emb = edges.src.pop('h'), edges.dst.pop('h')
    src_emb = poincare.log_map_zero(src_emb)
    dis_emb = poincare.log_map_zero(dis_emb)
    src_emb = src_emb.unsqueeze(dim=-2)
    dis_emb = dis_emb.unsqueeze(dim=-1)
    scores = torch.matmul(src_emb, dis_emb).squeeze(dim=1)
    return {'score': scores}

def hyperbolic_cosine(edges):
    # scores = torch.matmul(edges.src['h'].unsqueeze(dim=-2), edges.dst['h'].unsqueeze(dim=-1))
    from hyperbolic.PoincareManifold import PoincareManifold
    poincare = PoincareManifold(0, 0)
    src_emb, dis_emb = edges.src.pop('h'), edges.dst.pop('h')
    #src_emb = poincare.log_map_zero(src_emb)
    #dis_emb = poincare.log_map_zero(dis_emb)
    scores = torch.cosine_similarity(src_emb, dis_emb,dim=1).unsqueeze(dim=1)
    return {'score': scores}

def hyperbolic_cosine_2(edges):
    # scores = torch.matmul(edges.src['h'].unsqueeze(dim=-2), edges.dst['h'].unsqueeze(dim=-1))
    from hyperbolic.PoincareManifold import PoincareManifold
    poincare = PoincareManifold(0, 0)
    src_emb, dis_emb = edges.src.pop('h'), edges.dst.pop('h')
    #src_emb = poincare.log_map_zero(src_emb)
    #dis_emb = poincare.log_map_zero(dis_emb)
    cos = torch.cosine_similarity(src_emb, dis_emb,dim=1).unsqueeze(dim=1)
    #scores = 0.5 - 0.5 * cos - ((3 ** 0.5) * 0.5) * ((1 - cos**2 + 1e-6) ** 0.5)
    scores = (0.5 + cos) ** 2
    return {'score': scores}


