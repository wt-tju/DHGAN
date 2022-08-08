#import faiss
import numpy as np
import torch


def negative_sampler(g, nid, nid1, etype, negs_nums=1000):
    src_pos, dst = g.in_edges(nid, etype=etype)
    src_pos1, dst1 = g.in_edges(nid1, etype=etype)
    src_logits = torch.ones([g.num_nodes()])
    src_logits[src_pos] = 0
    src_logits[src_pos1] = 0
    src_logits[nid] = 0
    src_logits[nid1] = 0
    src_neg = src_logits.multinomial(negs_nums, replacement=True)
    return src_neg

def scores(ranks, target):
    def _hr(ranks, target):
        hr = 0
        for item in ranks:
            if item == target:
                hr = 1
        return hr
    def _mrr(ranks, target):
        mrr = 0
        for c, item in enumerate(ranks):
            if item == target:
                mrr = 1 / (c + 1)
                break
        return mrr

    def _ndcg(ranks, target):
        dcg = 0.0
        for c, item in enumerate(ranks):
            if item == target:
                dcg += 1.0 / np.log2(c + 2)
        idcg = 1.0
        ndcg = dcg / idcg
        return ndcg

    return _hr(ranks, target), _mrr(ranks, target), _ndcg(ranks, target)


#
# def faiss_topn(vector_data_array, topn=20):
#     row_num, dim = vector_data_array.shape
#     id_array = np.arange(row_num)
#     vector_data_query = vector_data_array
#
#     # quantizer = faiss.IndexFlatL2(dim)  # L2
#     quantizer = faiss.IndexFlatIP(dim)  # inner product
#     # index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
#     # index.train(vector_data_array)
#
#     quantizer.add(vector_data_array)  # add vectors to the index
#     D, I = quantizer.search(vector_data_query, topn)
#     return D, I
#
#
# def filter_topk(g_train, etype, index, topk=10):
#     # assert index.shape[0] == g_train.num_nodes()
#     index_topk = -np.ones([index.shape[0], topk], dtype=np.int64)
#     nids = g_train.ndata['_ID'].cpu().numpy()
#     for nid_subgraph in range(len(nids)):
#         u, v = g_train.in_edges(nid_subgraph, form='uv', etype=etype)
#         u = u.cpu().numpy()
#
#         nid = int(nids[nid_subgraph])
#         uids = nids[u]
#
#         idx_topk = 0
#         idx_ori = 0
#         while idx_topk < topk:
#             if not index[idx_ori] in uids:
#                 index_topk[nid][idx_topk] = index[nid][idx_ori]
#                 idx_topk += 1
#             idx_ori += 1
#     return index_topk

#
# def scores(g_test, etype, index):
#     nids = g_test.ndata['_ID'].cpu().numpy()
#     eids_subgraph = torch.arange(g_test.num_edges(), dtype=torch.int64)
#     src_subgraph, dst_subgraph = g_test.find_edges(eids_subgraph, etype=etype)
#     src_subgraph = src_subgraph.cpu().numpy()
#     dst_subgraph = dst_subgraph.cpu().numpy()
#     src_nids = nids[src_subgraph]
#     dst_nids = nids[dst_subgraph]
#
#     hr = _hr(index, src_nids, dst_nids)
#     mrr = _mrr(index, src_nids, dst_nids)
#     ndcg = _ndcg(index, src_nids, dst_nids)
#     return hr, mrr, ndcg
#
# def _hr(topk_index, src_nids, dst_nids):
#     hr = []
#     for idx in range(src_nids.shape[0]):
#         if src_nids[idx] in topk_index[dst_nids[idx]]:
#             hr.append(1)
#         else:
#             hr.append(0)
#     return sum(hr) / len(hr)
#
# def _mrr(topk_index, src_nids, dst_nids):
#     # reference: https://blog.csdn.net/guotong1988/article/details/98962159
#     mrr = []
#     for idx in range(src_nids.shape[0]):
#         mrr.append(0)
#         for rank, item in enumerate(topk_index[dst_nids[idx]]):
#             if item.item() == src_nids[idx].item():
#                 mrr[-1] = 1.0 / (rank + 1.0)
#                 break
#     return sum(mrr) / len(mrr)
#
# def _ndcg(topk_index, src_nids, dst_nids):
#     # reference:
#     # [1] https://www.cnblogs.com/by-dream/p/9403984.html
#     # [2] https://blog.csdn.net/guotong1988/article/details/98962159
#     ndcg = []
#     for idx in range(src_nids.shape[0]):
#         dcg = 0.0
#         for rank, item in enumerate(topk_index[dst_nids[idx]]):
#             if item.item() == src_nids[idx].item():
#                 dcg += 1.0 / np.log2(rank + 2)
#         idcg = 1.0
#         ndcg.append(dcg / idcg)
#     return sum(ndcg) / len(ndcg)
#
