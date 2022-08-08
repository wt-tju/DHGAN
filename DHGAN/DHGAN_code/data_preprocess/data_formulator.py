import json
import sys
import os
import torch
import dgl
import numpy as np


# def divide_dataset(g, threshold=5):
#     for etype in g.etypes:
#         g = dgl.remove_self_loop(g, etype=etype)
#     for etype in g.etypes:
#         train_mask = torch.tensor([False for i in range(g.num_edges(etype=etype))])
#         test_mask = torch.tensor([False for i in range(g.num_edges(etype=etype))])
#         for nid in range(g.num_nodes()):
#             u, v, eids = g.in_edges(nid, form='all', etype=etype)
#             eids = eids.cpu().numpy()
#             np.random.shuffle(eids)
#             eids_shuffle = torch.tensor(eids)
#             if len(eids_shuffle) < threshold:
#                 train_mask[eids_shuffle] = True
#             else:
#                 train_mask[eids_shuffle[:-1]] = True
#                 test_mask[eids_shuffle[-1:]] = True
#         g.edges[etype].data['train_mask'] = train_mask
#         g.edges[etype].data['test_mask'] = test_mask
#     return g


def divide_dataset(g, threshold=5):
    for etype in g.etypes:
        g = dgl.remove_self_loop(g, etype=etype)
    for etype in g.etypes:
        # train_mask = torch.tensor([False for i in range(g.num_edges(etype=etype))])
        test_mask = torch.tensor([False for i in range(g.num_edges(etype=etype))])
        choosed_eids = set()
        for nid in range(g.num_nodes()):
            eids = g.in_edges(nid, form='eid', etype=etype)
            eids = eids.cpu().numpy()
            np.random.shuffle(eids)
            eids_shuffle = torch.tensor(eids)
            if len(eids_shuffle) < threshold:
                pass
            else:
                u, v = g.find_edges(eids_shuffle[0], etype=etype)
                u, v = u.item(), v.item()
                if (not u in choosed_eids) and (not v in choosed_eids):
                    eid1 = g.edge_ids(u, v, etype=etype)
                    eid2 = g.edge_ids(v, u, etype=etype)
                    choosed_eids.add(u)
                    choosed_eids.add(v)
                    test_mask[eid1] = True
                    test_mask[eid2] = True
        train_mask = ~test_mask
        g.edges[etype].data['train_mask'] = train_mask
        g.edges[etype].data['test_mask'] = test_mask
        print('{}: | train edge:{}  test edge:{}'.format(etype, train_mask.sum().item(), test_mask.sum().item()))
    return g


def assign_num_id(u, node_to_num_id):
    if u not in node_to_num_id:
        node_to_num_id[u] = len(node_to_num_id)
    return node_to_num_id[u]


def init_node(num_id):
    return {
        "node_id": num_id,
        "node_type": 0,
        "node_weight": 1.0,
        "neighbor": {"0": [], "1": [], "2": [], "3": []},
        "uint64_feature": {},
        "edge": []
    }


def add_neighbor(u, v, edge_type, node_data):
    node_data[u]['neighbor'][str(edge_type)].append(v)


def add_edge(u, v, edge_type, node_data):
    uv_edge = {
        "src_id": u,
        "dst_id": v,
        "edge_type": edge_type,
        "weight": 1.0,
    }
    node_data[u]['edge'].append(uv_edge)


def fill_node_features(node_data, node_to_num_id, valid_node_asin):
    def id_mapping(s, id_dict):
        if s not in id_dict:
            id_dict[s] = len(id_dict)
        return id_dict[s]

    cid1_dict, cid2_dict, cid3_dict, bid_dict = {}, {}, {}, {}
    node_features = {}

    for eachline in meta_file:
        data = eval(eachline)
        asin = data['asin']
        if asin in valid_node_asin:
            c1, c2, c3 = data['category'][:3]
            if not 'brand' in data:
                brand = 'none'
            else:
                brand = data['brand']
            cid1, cid2, cid3 = id_mapping(c1, cid1_dict), id_mapping(c2, cid2_dict), \
                               id_mapping(c3, cid3_dict)
            bid = id_mapping(brand, bid_dict)

            num_id = node_to_num_id[asin]
            node_data[num_id]['uint64_feature'] = {"0": [cid2], "1": [cid3], "2": [bid]}

    feature_stats = "#cid2: {}; #cid3: {}; #bid: {}".format(len(cid2_dict), len(cid3_dict), len(bid_dict))
    print(feature_stats)
    log_file.write(feature_stats + '\n')


def main():
    print("Converting node data to json format...")

    node_to_num_id = {}
    node_data = {}

    valid_node_asin = set()

    for eachline in sim_edges_file:
        u, v, w = eachline.strip('\n').split('\t')
        valid_node_asin.add(u)
        valid_node_asin.add(v)
        uid = assign_num_id(u, node_to_num_id)
        vid = assign_num_id(v, node_to_num_id)

        if uid not in node_data:
            node_data[uid] = init_node(uid)
        if vid not in node_data:
            node_data[vid] = init_node(vid)

        add_neighbor(uid, vid, 0, node_data)
        if uid != vid:
            add_neighbor(vid, uid, 0, node_data)

    for eachline in cor_edges_file:
        u, v, w = eachline.strip('\n').split('\t')
        valid_node_asin.add(u)
        valid_node_asin.add(v)
        uid = assign_num_id(u, node_to_num_id)
        vid = assign_num_id(v, node_to_num_id)

        if uid not in node_data:
            node_data[uid] = init_node(uid)
        if vid not in node_data:
            node_data[vid] = init_node(vid)

        add_neighbor(uid, vid, 1, node_data)
        if uid != vid:
            add_neighbor(vid, uid, 1, node_data)

    fill_node_features(node_data, node_to_num_id, valid_node_asin)

    total_sim_num, total_cor_num = 0, 0
    for u in sorted(node_data.keys()):
        u_data = node_data[u]
        u_neighbor = node_data[u]['neighbor']
        total_sim_num += len(u_neighbor['0'])
        total_cor_num += len(u_neighbor['1'])

        for edge_type in [0, 1]:
            for v in u_data['neighbor'][str(edge_type)]:
                uid, vid = u, v
                add_edge(uid, vid, edge_type, node_data)
            # u_data['neighbor'][str(edge_type)] = {
            #     v: 1.0 for v in u_data['neighbor'][str(edge_type)]}
        galileo_form_heterogeneous1_0(u_data)
        # print(u_data)
        # json.dump(u_data, out_file)
        # out_file.write('\n')
    galileo_outfile_node.close()
    galileo_outfile_edge.close()
    print('galileo form file finished!')

    g = dgl_form_heterogeneous()
    print('dgl form file finished!')
    info = 'Total node num: {}\nTotal edge num: {}; sim: {}; cor: {}\n'.\
          format(g.num_nodes('item'), g.num_edges('sim') + g.num_edges('cor'), g.num_edges('sim'),
                 g.num_edges('cor'))
    data_stats = "dgl dataset-{}:\n{}\n".format(data_name, info)

    print(data_stats)
    log_file.write(data_stats)

    for asin, num_id in node_to_num_id.items():
        id_dict_file.write("{}\t{}\n".format(asin, num_id))



def galileo_form_heterogeneous1_0(node_data):
    if len(node_data['edge']) == 0:
        print(node_data)
        return
    # node: id:int;type_0:int;weight:float;f1:int;f2:int;f3:int
    node_info = '{};{};{};{};{};{}\n'.format(int(node_data['node_id']),
                                             int(node_data['node_type']),
                                             float(node_data['node_weight']),
                                             int(node_data['uint64_feature']['0'][0]),
                                             int(node_data['uint64_feature']['1'][0]),
                                             int(node_data['uint64_feature']['2'][0]))
    galileo_outfile_node.write(node_info)

    # edge: src:int;dst:int;type_0:int;weight:float
    for e in node_data['edge']:
        edge_info = '{};{};{};{}\n'.format(int(e['src_id']),
                                           int(e['dst_id']),
                                           int(e['edge_type']),
                                           float(e['weight']))
        galileo_outfile_edge.write(edge_info)


def dgl_form_heterogeneous():
    node_path = './galileo/{}_all_node_info'.format(data_name)
    edge_path = './galileo/{}_all_edge_info'.format(data_name)
    metapath = [('item', 'sim', 'item'), ('item', 'cor', 'item')]
    data_dict = {
        metapath[0]: [],
        metapath[1]: [],
    }
    with open(edge_path, 'r') as f:
        for eachline in f:
            eachline = eachline[:-1].split(';')
            src, dst, type = eachline[:3]
            if type == '0':
                data_dict[metapath[0]].append(torch.tensor([int(src), int(dst)]))
            else:
                data_dict[metapath[1]].append(torch.tensor([int(src), int(dst)]))
    g = dgl.heterograph(data_dict=data_dict)

    with open(node_path, 'r') as f:
        feats = [None for i in range(g.num_nodes('item'))]
        for eachline in f:
            eachline = eachline[:-1].split(';')
            node_id, type, weight, f1, f2, f3 = eachline
            node_id = int(node_id)
            f1 = int(f1)
            f2 = int(f2)
            f3 = int(f3)
            feats[node_id] = [f1, f2, f3]
        feats = torch.tensor(feats)
        print(feats.shape)
    g.nodes['item'].data['sparse_feats'] = feats

    g = divide_dataset(g)
    print('Total node num: {}\nTotal edge num: {}; sim: {}; cor: {}'.
          format(g.num_nodes('item'), g.num_edges('sim') + g.num_edges('cor'), g.num_edges('sim'), g.num_edges('cor')))

    save_path = os.path.join('../graph_datasets')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    dgl.save_graphs(os.path.join(save_path, '{}.graph'.format(data_name)), g)
    return g



if __name__ == '__main__':
    # data_name = 'Clothing_Shoes_and_Jewelry'
    data_name = 'Electronics'
    # data_name = sys.argv[1]
    raw_datapath = "./raw_data/meta_{}.json".format(data_name)
    meta_file = open(raw_datapath).readlines()

    sim_filename = "{}_sim.edges".format(data_name)
    cor_filename = "{}_cor.edges".format(data_name)

    sim_edges_file = open("./tmp/" + sim_filename, 'r')
    cor_edges_file = open("./tmp/" + cor_filename, 'r')

    log_file = open("./stats/{}.log".format(data_name), 'w')
    id_dict_file = open('./tmp/{}_id_dict.txt'.format(data_name), 'w')

    if not os.path.exists('./galileo/'.format(data_name)):
        os.makedirs('./galileo/'.format(data_name))

    galileo_outfile_node = open('./galileo/{}_all_node_info'.format(data_name), 'w')
    galileo_outfile_edge = open('./galileo/{}_all_edge_info'.format(data_name), 'w')

    main()

