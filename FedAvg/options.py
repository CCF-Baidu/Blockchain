#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=50, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=20, help="number of users: K")
    parser.add_argument('--num_attackers', type=int, default=30, help="number of attackers: f")
    parser.add_argument('--frac', type=float, default=1, help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B")
    parser.add_argument('--virtual_batch_size', type=int, default=64, help="virtual_batch_size")
    parser.add_argument('--local_iter', type=int, default=30, help="local iteration(number of batch)")
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.5)')

    # sampling arguments
    parser.add_argument('--single', action='store_true', help="assign single digits to each user")
    parser.add_argument('--fix_total', action='store_true', help="fix total users to 100")

    # model arguments
    parser.add_argument('--model', type=str, default='smallcnn', help='model name')
    parser.add_argument('--mu', type=int, default=1, help="mu of fedprox")
    parser.add_argument('--algorithm', default='fedprox', help="algorithm_fedprox")

    # DP arguments
    parser.add_argument('--testDP', type=bool, default=False, help="use backdoor attack")
    parser.add_argument('--epsilon', type=float, default=2, help='sampling_prob')
    parser.add_argument('--max_grad_norm', type=float, default=1, help='DP MAX_GRAD_NORM')
    parser.add_argument('--noise_multiplier', type=float, default=2, help='DP NOISE_MULTIPLIER')
    parser.add_argument('--delta', type=float, default=.00001, help='DP DELTA')
    parser.add_argument('--sampling_prob', type=int, default=0.03425, help='sampling_prob')
    parser.add_argument('--guass', type=bool, default=True, help="use guass")
    # attack arguments
    parser.add_argument('--attackertype', type=str, default='backdoor', choices=['signflipping', 'samevalue', 'backdoor'], help="attack type")
    parser.add_argument('--use_normal', type=int, default=0, help='perform gaussian attack on n users')
    parser.add_argument('--normal_scale', type=float, default=1.0, help='scale of noise in percent')
    parser.add_argument('--attacker_ep', type=int, default=10, help="the number of attacker's local epochs: E")
    parser.add_argument('--change_rate', type=float, default=-1.0, help='scale of noise in percent')
    parser.add_argument('--use_poison', type=int, default=-1, help='perform poison attack on n users')
    parser.add_argument('--attack_label', type=int, default=-1, help='select the label to be attacked in poisoning attack')
    parser.add_argument('--donth_attack', default=-1, action='store_true', help='this attack excludes the selected nodes from aggregation')
    # backdoor attack arguments
    parser.add_argument('--lpdp', type=bool, default=False, help="use lp")
    parser.add_argument('--is_backdoor',type=bool, default=True, help="use backdoor attack")
    parser.add_argument('--backdoor_per_batch',type=int,default=20, help="poisoned data during training per batch")
    parser.add_argument('--backdoor_scale_factor', type=float, default=1.0, help="scale factor for local model's parameters")

    parser.add_argument('--backdoor_label', type=int, default=2, help="target label for backdoor attack")
    parser.add_argument('--backdoor_single_shot_scale_epoch', type=int, default=-1, help="used for one-single-shot; -1 means no single scaled shot")

    # aggregation arguments
    parser.add_argument('--agg', type=str, default='ours', choices=['improve_irls','average','ours','sign_irls', 'median', 'trimmed_mean',
                                                                           'repeated', 'irls', 'simple_irls',
                                                                           'irls_median', 'irls_theilsen',
                                                                           'irls_gaussian', 'fg'],
                        help="Aggregation methods")
    parser.add_argument('--cloudagg', type=str, default='average', choices=['average', 'sign', 'sign_irls', 'irls'],
                        help="Cloud aggregation methods")
    parser.add_argument('--gamma1', type=float, default = 0.015,
                        help="sign aggregation arguments")
    parser.add_argument('--gamma2', type=float, default=0.01,
                        help="sign aggregation arguments")
    parser.add_argument('--Lambda', type=float, default=2.0, help='set lambda of irls (default: 2.0)')
    parser.add_argument('--thresh', type=float, default=0.1, help='set thresh of irls restriction (default: 0.1)')
    parser.add_argument('--alpha', type=float, default=0.2, help='set thresh of trimmed mean (default: 0.2)')
    parser.add_argument('--use_memory', type=str2bool, default=True, help="use FoolsGold memory option")
    parser.add_argument('--alpe', type=float, default=1, help='alpe')
    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', type=int, default=0, help='whether i.i.d or not, 1 for iid, 0 for non-iid')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=-1, help="GPU ID")
    parser.add_argument('--verbose', type=int, default=0, help='verbose print, 1 for True, 0 for False')
    parser.add_argument('--seed', type=int, default=1234, help='random seed (default: 1234)')

    # MEC arguments
    parser.add_argument('--num_edge_aggregation',type=int,default=1,help='number of edge aggregation (tau_2)')
    parser.add_argument('--edgeiid',type=int,default=1,help='distribution of the data under edges, 1 (edgeiid),0 (edgeniid) (used only when iid = -2)')
    parser.add_argument('--num_edges',type=int,default=5,help='number of edges')

    args = parser.parse_args()
    return args
