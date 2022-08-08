import argparse
import os
import torch
from executor import Executor
import warnings

parser = argparse.ArgumentParser(description='???')

# parser.add_argument('--model', type=str, required=False, default='decgcn', help='model of experiment, options: []')
parser.add_argument('--model', type=str, required=False, default='hyperbolic', help='model of experiment, options: []')
parser.add_argument('--model_name', type=str, required=False, default='hyperbolic', help='model of experiment, options: []')
#parser.add_argument('--dataset', type=str, required=False, default='Electronics', help='dataset')
parser.add_argument('--dataset', type=str, required=False, default='Toys_and_Games', help='dataset')
parser.add_argument('--graph_root_path', type=str, default='./graph_datasets/', help='root path of the graph data file')
parser.add_argument('--graph_data_file', type=str, default='', help='graph data file')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--embedding_save_path', type=str, default='./save_embedding/', help='embedding save path')
parser.add_argument('--log_path', type=str, default='./logs/', help='log path')


parser.add_argument('--use_nodeid_emb', type=bool, default=True, help='whether use node embedding')
parser.add_argument('--nodeid_emb_dim', type=int, default=10, help='node embedding dims')
parser.add_argument('--sparse_maxs', type=str, default='auto', help='sparse features max values array')
parser.add_argument('--sparse_emb_dim', type=str, default='auto', help='sparse features embedding dims')
parser.add_argument('--hidden_dim', type=int, default=64, help='numbers of hidden embedding dims')
parser.add_argument('--output_dim', type=int, default=32, help='numbers of hidden embedding dims')
parser.add_argument('--negs_nums', type=int, default=5, help='numbers of negative samples')
parser.add_argument('--num_layers', type=int, default=2, help='numbers of GNN layers')
parser.add_argument('--fanout', type=str, default='10,10', help='numbers of fanout') # number of sampled neighbor
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--remove_ratio', type=float, default=0.1, help='remove edges')

parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--train_epochs', type=int, default=2, help='train epochs')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size of train input data')
parser.add_argument('--learning_rate', type=float, default=0.01, help='optimizer learning rate')
parser.add_argument('--run_mode', type=str, default='train', help='train or save embedding')
parser.add_argument('--device', type=str, default='cpu', help='device')

parser.add_argument('--embedding_use_model', type=str, default='epoch1', help='embedding use the model name')
parser.add_argument('--eval_neg_nums', type=int, default=1000, help='evaluate negative sampling numbers')
parser.add_argument('--eval_topk', type=int, default=10, help='evaluate top k')

args = parser.parse_args()

def args_init():
    args.graph_data_file = '{}.graph'.format(args.dataset)
    args.fanout = [int(x) for x in args.fanout.split(',')]
    #args.fanout = [args.fanout[0]]
    if args.model == 'gatne':
        args.fanout = [args.fanout[0]]


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    args_init()
    opt = Executor(args)  ## 
    print('Dataset: {} | Model: {}'.format(args.dataset, args.model))
    if args.run_mode == 'train':
        opt.train()  ## train mode
    elif args.run_mode == 'save_emb':
        opt.save_embedding()
    elif args.run_mode == 'eval':
        opt.eval()

