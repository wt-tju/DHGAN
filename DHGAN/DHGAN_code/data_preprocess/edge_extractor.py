import sys
from collections import defaultdict


def main():
    f = open(raw_datapath, 'r').readlines()
    sim_edges = defaultdict(int)
    rel_edges = defaultdict(int)

    print("Filtering items with not sub/comp edges...")

    all_asin = set()
    for eachline in f:
        each_data = eval(eachline)
        if not 'also_buy' in each_data:
            each_data['also_buy'] = []
        if not 'also_view' in each_data:
            each_data['also_view'] = []
        # print(each_data.keys())
        # print(each_data['category'], len(each_data['category']))
        if len(each_data['category']) >= 3 and len(each_data['also_buy']) + len(each_data['also_view']) > 0:
            asin = each_data['asin']
            all_asin.add(asin)

    for asin in all_asin:
        asin = str(asin)
        sim_edges[(asin, asin)] += 1
        rel_edges[(asin, asin)] += 1

    for eachline in f:
        each_data = eval(eachline)
        asin = each_data['asin']
        if asin not in all_asin:
            continue
        related_data = each_data

        if 'also_view' in related_data:
            for rid in related_data['also_view']:
                if rid not in all_asin:
                    continue
                u, v = str(asin), str(rid)
                if u > v: u, v = v, u
                edge = (u, v)
                sim_edges[edge] += 1

        if 'also_buy' in related_data:
            for rid in related_data['also_buy']:
                if rid not in all_asin:
                    continue
                u, v = str(asin), str(rid)
                if u > v: u, v = v, u
                edge = (u, v)
                rel_edges[edge] += 1

    fout_sim = open("./tmp/{}_sim.edges".format(data_name), 'w')
    fout_rel = open("./tmp/{}_cor.edges".format(data_name), 'w')

    max_sim_weight = 0
    max_rel_weight = 0
    all_nodes = set()
    for (u, v), w in sim_edges.items():
        # assert u <= v, '{}, {}'.format(u, v)
        fout_sim.write('\t'.join([u, v, str(w)]) + '\n')
        max_sim_weight = max(max_sim_weight, w)
        all_nodes.add(u)
        all_nodes.add(v)
    for (u, v), w in rel_edges.items():
        # assert u <= v, '{}, {}'.format(u, v)
        fout_rel.write('\t'.join([u, v, str(w)]) + '\n')
        max_rel_weight = max(max_rel_weight, w)
        all_nodes.add(u)
        all_nodes.add(v)

    print("max_sim_weight: {}; max_rel_weight: {}; node num: {}".format(
        max_sim_weight, max_rel_weight, len(all_nodes)))


if __name__ == "__main__":
    data_name = sys.argv[1]
    # data_name = 'Clothing_Shoes_and_Jewelry'
    raw_datapath = "./raw_data/meta_{}.json".format(data_name)
    main()

