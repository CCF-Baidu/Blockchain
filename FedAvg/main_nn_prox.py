#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
import warnings
warnings.filterwarnings('ignore')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import copy
import numpy as np
from tqdm import tqdm
import torch
from datasets import build_datasets
from tensorboardX import SummaryWriter
from options import args_parser
from Update import LocalUpdate
from FedNets import build_model
from averaging import aggregate_weights, get_valid_models, FoolsGold, IRLS_aggregation_split_restricted, average_weights, aggregate_weights_cloud
from attack import add_gaussian_noise, change_weight
import statistics
from torch.utils.data import Subset
import math
from utils import test_inference
def WriteToFile(args,backdooracc,acc,loss):
    with open('resultC.txt', 'a+') as fd:
        if args.is_backdoor:
            fd.write("method:" + args.agg + '\n' + "attackertype:"+ args.attackertype + 'DP MAX_GRAD_NORM:' + str(args.max_grad_norm) + 'DP NOISE_MULTIPLIER:' + str(args.noise_multiplier) + '\n' + "attackernum:" + str(args.num_attackers) + '\n' + "backdooracc:" + str(backdooracc) + '\n' + "acc:" + str(acc) + '\n' + "loss:" + str(loss) + '\n' )
            fd.close()
        else:
            fd.write("method:" + args.agg + "attackertype:"+ args.attackertype + "attackernum:" + str(args.num_attackers) + 'DP MAX_GRAD_NORM:' + str(args.max_grad_norm) + 'DP NOISE_MULTIPLIER:' + str(args.noise_multiplier) + '\n'  + "acc:" + str(acc) + '\n' + "loss:" + str(loss) + '\n' )
            fd.close()
def test(net_g, dataset, args, dict_users):
    # testing
    list_acc = []
    list_loss = []
    net_test = copy.deepcopy(net_g)
    net_test.eval()
    if args.dataset == "mnist":
        classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    elif args.dataset == "fmnist":
        classes = ('t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle-boot')
    elif args.dataset == "cifar":
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    elif args.dataset == "loan":
        classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8')
    with torch.no_grad():
        added_acc=0
        added_loss=0
        added_data_num=0
        for c in range(len(classes)):
            if c < len(classes) and len(dict_users[c])>0:

                net_local = LocalUpdate(args=args, dataset=dataset, idxs=dict_users[c], tb=None, test_flag=True)
                acc, loss = net_local.test(net=net_test)
                # print("test accuracy for label {} is {:.2f}%".format(classes[c], acc * 100.))
                list_acc.append(acc)
                list_loss.append(loss)
                added_acc+= acc*len(dict_users[c])
                added_loss+= loss*len(dict_users[c])
                added_data_num+=len(dict_users[c])

        print("average acc: {:.2f}%,\ttest loss: {:.5f}".format(100. * added_acc/ float(added_data_num),
                                                                added_loss/ float(added_data_num)))

    return list_acc


def backdoor_test(net_g, dataset, args, idxs):
    # backdoor testing
    net_test = copy.deepcopy(net_g)
    net_test.eval()
    with torch.no_grad():
        net_local = LocalUpdate(args=args, dataset=dataset, idxs=idxs, tb=None,
                                backdoor_label=args.backdoor_label, test_flag=True)
        acc, loss = net_local.backdoor_test(net=net_test)
        print("backdoor acc: {:.2f}%,\ttest loss: {:.5f}".format(100. * acc, loss))
    return acc


if __name__ == '__main__':
    # parse args
    args = args_parser()
    np.random.seed(args.seed)
    learning_rate = args.lr
    # set attack mode
    print('perform poison attack with {} attackers'.format(args.num_attackers))

    # define paths
    path_project = os.path.abspath('..')
    if args.gpu != -1:
        torch.cuda.set_device(args.gpu)

    summary = SummaryWriter('local')

    dataset_train, dataset_test, dict_users, test_users, attackers = build_datasets(args)
    va_index = np.random.choice(range(len(dataset_test)), 500)  # 划分验证集，从测试集中随机选择500个
    dataset_v = Subset(dataset_test, va_index)
    if not args.fix_total:
        args.num_users += args.num_attackers

    # check poison attack
    if args.dataset == 'mnist' and args.num_attackers > 0:
        assert args.attack_label == 1 or (args.donth_attack and args.attack_label < 0)
    if args.dataset == 'fmnist' and args.num_attackers > 0:
        assert args.attack_label == 1 or (args.donth_attack and args.attack_label < 0)
    elif args.dataset == 'cifar' and args.num_attackers > 0:
        assert args.attack_label == 3 or (args.donth_attack and args.attack_label < 0)
    elif args.dataset == 'loan' and args.num_attackers > 0:
        assert args.attack_label == 0

    # build model
    if args.gpu != -1:
        torch.cuda.set_device(args.gpu)
    net_glob = build_model(args)

    print(net_glob)

    # init FoolsGold
    if args.agg == 'fg':
        fg = FoolsGold(args)
    else:
        fg = None

    # copy weights
    w_glob = net_glob.state_dict()
    w_0 = copy.deepcopy(w_glob)
    net_glob.train()

    ######################################################
    # Training                                           #
    ######################################################
    loss_train = []
    accs = []
    reweights = []
    backdoor_accs = []
    if 'irls' in args.agg:
        model_save_path = '../weights/{}_{}_irls_{}_{}_{}'.format(args.dataset, args.model, args.Lambda, args.thresh, args.iid)
    else:
        model_save_path = '../weights/{}_{}'.format(args.dataset, args.model)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    last_bd_acc=0
    lr_change_ratio=10.0
    # MEC选取客户端
    user_total = range(args.num_users)
    users_per_edge = int(args.num_users / args.num_edges)
    edge_user_list = []
    for i in range(args.num_edges):
        # Randomly select clients and assign them
        selected_users = np.random.choice(user_total, users_per_edge, replace=False)
        user_total = list(set(user_total) - set(selected_users))
        edge_user_list.append(selected_users)
    print(edge_user_list)

    train_loss_plot = []
    test_acc_plot = []
    empty = []
    for iter in tqdm(range(args.epochs)):
        ones_tensor = copy.deepcopy(w_glob)
        zeros_tensor = copy.deepcopy(w_glob)
        negativeone_tensor = copy.deepcopy(w_glob)
        uniform_tensor = copy.deepcopy(w_glob)
        for lk in w_glob.keys():
            ones_tensor[lk] = torch.ones_like(w_glob[lk])
            zeros_tensor[lk] = torch.zeros_like(w_glob[lk])
            negativeone_tensor[lk] = torch.zeros_like(w_glob[lk])-torch.ones_like(w_glob[lk])
            uniform_tensor[lk] = torch.zeros_like(w_glob[lk]).uniform_(-1, 1)
        # if iter>=1:
        #     args.lr = args.lr * 0.999
        np.random.seed(iter + 1)
        print('Epoch:', iter, "/", args.epochs)
        net_glob.train()
        w_noise = copy.deepcopy(w_glob)
        sign_aggwight = copy.deepcopy(w_glob)
        for lk in w_glob.keys():
            sign_aggwight[lk] = torch.zeros_like(w_glob[lk])
        sign_localwight = copy.deepcopy(sign_aggwight)
        w_signinit = copy.deepcopy(w_glob)
        edge_weights = []
        edge_loss = []
        edge_cloudweights = []
        for edge in range(args.num_edges):
            w_locals, loss_locals = [], []
            m = max(int(args.frac * len(edge_user_list[edge])), 1)
            # print('taking {} users'.format(m))
            idxs_users = np.random.choice(edge_user_list[edge], m, replace=False, p=None)
            sample_user = idxs_users.tolist()  # 随机选取m个竞争者，将user编号数组转化为列表
            print('sample client={}'.format(idxs_users))
            distance_client = []  # 存放各客户端模型与初始模型的距离
            acc_comps = []
            for idx in idxs_users:
                # print(idx)
                if (idx >= args.num_users - args.num_attackers and not args.fix_total) or \
                        (args.fix_total and idx in attackers):
                    print('{} is a attackclient'.format(idx))
                    local_ep = args.local_ep
                    if args.attacker_ep != args.local_ep:
                        if args.dataset == 'loan':
                            lr_change_ratio = 1.0/5
                        args.lr = args.lr * lr_change_ratio
                    args.local_ep = args.attacker_ep
                    if args.is_backdoor:
                        if args.backdoor_single_shot_scale_epoch ==-1 or iter == args.backdoor_single_shot_scale_epoch:
                            print('backdoor attacker', idx)
                            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], tb=summary,
                                            backdoor_label=args.backdoor_label)
                        else:
                            args.local_ep = local_ep
                            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], tb=summary)
                    else:
                        # # if args.fix_total and args.attack_label < 0 and args.donth_attack:
                        # #     continue
                        # else:
                        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], tb=summary,
                                            attack_label=args.attack_label)

                    temp_net = copy.deepcopy(net_glob)
                    if args.agg == 'fg':
                        w, loss, _poisoned_net = local.u                                                                                                                                                                                                                                                                                                                                                                         pdate_gradients(net=temp_net)
                    else:
                        w, loss, _poisoned_net = local.update_weights_fedprox(net=temp_net)
                    if args.attackertype == 'signflipping':
                        if args.lpdp == True:
                            for lk in w_glob.keys():
                                # add gaussian noise
                                # w_noise[lk] = torch.normal(0, noised_sigma, w_noise[lk].size())
                                # add laplace noise
                                # para_max = noised_w[lk].tolist()
                                noised_sigma = float(2 * torch.max(torch.abs(w[lk], out=None)) * args.local_ep) / float(
                                    args.epochs * args.epsilon)
                                distribution_laplace = torch.distributions.laplace.Laplace(0.0, noised_sigma)
                                w_noise[lk] = distribution_laplace.sample(w_noise[lk].size())
                                w[lk] = w[lk] + w_noise[lk]
                                w[lk] = -10 * w[lk]  # sign_flipping_attacks
                    elif args.attackertype == 'samevalue':
                        for k in w.keys():
                            w[k] = 100 * torch.ones(w[k].shape)  # same_value
                    else:
                        # change a portion of the model gradients to honest
                        if 0 < args.change_rate < 1.:
                            w_honest, reweight = IRLS_aggregation_split_restricted(w_locals, args.Lambda, args.thresh)
                            w = change_weight(w, w_honest, change_rate=args.change_rate)
                    args.local_ep = local_ep
                    if args.attacker_ep != args.local_ep:
                        args.lr = args.lr / lr_change_ratio
                else:
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], tb=summary)
                    temp_net = copy.deepcopy(net_glob)
                    if args.agg == 'fg':
                        w, loss, _updated_net = local.update_gradients(net=temp_net)
                    else:
                        w, loss, _updated_net = local.update_weights_fedprox(net=temp_net)
                    if args.lpdp == True:
                        for lk in w_glob.keys():
                            # add gaussian noise
                            # w_noise[lk] = torch.normal(0, noised_sigma, w_noise[lk].size())
                            # add laplace noise
                            # para_max = noised_w[lk].tolist()
                            noised_sigma = float(2 * torch.max(torch.abs(w[lk], out=None)) * args.local_ep) / float(
                                args.epochs * args.epsilon)
                            distribution_laplace = torch.distributions.laplace.Laplace(0.0, noised_sigma)
                            w_noise[lk] = distribution_laplace.sample(w_noise[lk].size())
                            w[lk] = w[lk] + w_noise[lk]
                    # if args.guass == True:
                    #     sigma = math.sqrt(2.0 * math.log(1.25 / args.delta)) / (args.epsilon * args.epochs)
                    #     for lk in w_glob.keys():
                    #         #w[lk] = torch.nn.utils.clip_grad_norm(w[lk], args.max_grad_norm, norm_type=2)
                    #         distribution_guass = torch.distributions.Normal(0.0, sigma * args.max_grad_norm)
                    #         w[lk] = w[lk] + distribution_guass.sample(w_noise[lk].size())
                if args.agg == 'sign_irls':
                    w_sign = copy.deepcopy(w)
                    sysi_w = copy.deepcopy(w)
                    worngkey = 0
                    for lk in w.keys():
                        # w[lk] = torch.where(w[lk] == 0, uniform_tensor[lk], w_int[lk])
                        #w_sign[lk] = torch.where(w[lk] - w_signinit[lk] <= 0, negativeone_tensor[lk], ones_tensor[lk])
                        sysi_w[lk] = torch.sign(w[lk].mul(w_signinit[lk]))
                        sysi_w[lk] = torch.where(sysi_w[lk] == 0, torch.sign(uniform_tensor[lk]), sysi_w[lk])
                        if torch.median(sysi_w[lk])<=0:
                            worngkey = worngkey + 1
                    if worngkey/len(w.keys())>0.5:
                        direction = -1
                    else:
                        direction = 1
                    for lk in w.keys():
                        # w[lk] = torch.where(w[lk] == 0, uniform_tensor[lk], w_int[lk])
                        #w_sign[lk] = torch.where(w[lk] - w_signinit[lk] <= 0, negativeone_tensor[lk], ones_tensor[lk])
                        w_sign[lk] =  torch.sign(direction * w[lk] - w_signinit[lk])#torch.mean(torch.sign(w[lk].mul(w_signinit[lk]))) *
                        w_sign[lk] = torch.where(w_sign[lk] == 0, torch.sign(uniform_tensor[lk]), w_sign[lk])
                        #w_sign[lk] = direction * w_sign[lk]
                    w_locals.append(copy.deepcopy(w_sign))
                elif args.agg == 'ours':
                    w_d = copy.deepcopy(w)
                    sy_w = copy.deepcopy(w)
                    distance = 0
                    worngkey = 0
                    for lk in w_d.keys():
                        sy_w[lk] = torch.sign(w[lk].mul(w_glob[lk]))
                        sy_w[lk] = torch.where(sy_w[lk] == 0, torch.sign(uniform_tensor[lk]), sy_w[lk])
                        if torch.median(sy_w[lk]) < 0:
                            worngkey = worngkey + 1
                    if worngkey / len(w.keys()) > 0.5:
                        direction = -1
                    else:
                        direction = 1
                        # div1 = torch.median(
                        #     torch.abs(torch.addcdiv(torch.zeros_like(w[k]), 1, w[k], w_glob[k], out=None)))
                        # div2 = torch.median(
                        #     torch.abs(torch.addcdiv(torch.zeros_like(w[k]), 1, w_glob[k], w[k], out=None)))
                        # direction = torch.sign(w[k].mul(w_glob[k]))
                    if direction == 1:
                        for k in w.keys():
                            distance += (w[k]- w_glob[k]).norm(2)
                    else:
                        distance = torch.tensor(float("inf"))
                    # if distance < 0:
                    #     distance = torch.tensor(float("inf"))
                    distance_client.append(distance.item())
                    net_v = copy.deepcopy(net_glob)
                    net_v.load_state_dict(w)
                    # dataset_v = Subset(dataset_v,
                    #                np.array(list(set(range(len(dataset_v))).difference(empty))))  # 测试集-验证集=最终的测试集
                    acc_test, loss_test = test_inference(args, net_v, dataset_v)
                    acc_comps.append(acc_test)
                    score = (1 + args.alpe / (iter + 1)) * acc_test * 100 - distance
                    print('score={},distance={},acc={},direction={}'.format(score,distance,acc_test,direction))
                    if score >= 0:
                        w_locals.append(copy.deepcopy(w))
                elif args.agg == 'sign':
                    w_int = copy.deepcopy(w)
                    sy_w = copy.deepcopy(w)
                    worngkey = 0
                    for lk in w.keys():
                        # w[lk] = torch.where(w[lk] == 0, uniform_tensor[lk], w_int[lk])
                        # w_sign[lk] = torch.where(w[lk] - w_signinit[lk] <= 0, negativeone_tensor[lk], ones_tensor[lk])
                        sy_w[lk] = torch.sign(w[lk].mul(w_signinit[lk]))
                        sy_w[lk] = torch.where(sy_w[lk] == 0, torch.sign(uniform_tensor[lk]), sy_w[lk])
                        if torch.median(sy_w[lk]) <= 0:
                            worngkey = worngkey + 1
                    if worngkey / len(w.keys()) > 0.5:
                        direction = -1
                    else:
                        direction = 1
                    for lk in w.keys():
                        # w[lk] = torch.where(w[lk] == 0, uniform_tensor[lk], w_int[lk])
                        #w_sign[lk] = torch.where(w[lk] - w_signinit[lk] <= 0, negativeone_tensor[lk], ones_tensor[lk])
                        w_int[lk] = torch.sign(direction * w[lk] - w_signinit[lk])
                        w_int[lk] = torch.where(w_int[lk] == 0, torch.sign(uniform_tensor[lk]), w_int[lk])
                        sign_localwight[lk] = sign_localwight[lk] + w_int[lk] #torch.mean(torch.sign(w[lk].mul(w_signinit[lk]))) *
                elif args.agg == 'improve_irls':
                    sy_w = copy.deepcopy(w)
                    worngkey = 0
                    for lk in w.keys():
                        # w[lk] = torch.where(w[lk] == 0, uniform_tensor[lk], w_int[lk])
                        # w_sign[lk] = torch.where(w[lk] - w_signinit[lk] <= 0, negativeone_tensor[lk], ones_tensor[lk])
                        sy_w[lk] = torch.sign(w[lk].mul(w_signinit[lk]))
                        sy_w[lk] = torch.where(sy_w[lk] == 0, torch.sign(uniform_tensor[lk]), sy_w[lk])
                        if torch.median(sy_w[lk]) <= 0:
                            worngkey = worngkey + 1
                    if worngkey / len(w.keys()) > 0.5:
                        direction = -1
                    else:
                        direction = 1
                    for lk in w.keys():
                        w[lk] = direction * w[lk]
                    w_locals.append(copy.deepcopy(w))
                else:
                    w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))
            # remove model with inf values
            w_locals, invalid_model_idx= get_valid_models(w_locals)
            # if len(w_locals) == 0:
            #     continue
            if args.agg == 'sign_irls':
                edge_aggwight = aggregate_weights(args, w_locals, net_glob, reweights, fg)
                w_eint = copy.deepcopy(edge_aggwight)
                for lk in edge_aggwight.keys():
                    #edge_aggwight[lk] = w_signinit[lk] + args.gamma * torch.where(edge_aggwight[lk] <= 0, negativeone_tensor[lk], ones_tensor[lk])
                    w_eint[lk] = torch.sign(edge_aggwight[lk])
                    w_eint[lk] = torch.where(w_eint[lk] == 0, torch.sign(uniform_tensor[lk]), w_eint[lk])
                    edge_aggwight[lk] = w_signinit[lk] + args.gamma1 * w_eint[lk]
            elif args.agg == 'sign':
                edge_aggwight = copy.deepcopy(w_signinit)
                w_seint = copy.deepcopy(w_signinit)
                for lk in w_signinit.keys():
                    w_seint[lk] = torch.sign(sign_localwight[lk])
                    w_seint[lk] = torch.where(w_seint[lk] == 0, torch.sign(uniform_tensor[lk]), w_seint[lk])
                    edge_aggwight[lk] = w_signinit[lk] + args.gamma1 * w_seint[lk]
            else:
                print(len(w_locals))
                edge_aggwight = aggregate_weights(args, w_locals, net_glob, reweights, fg)
            # for lk in edge_aggwight.keys():
            #     sign_aggwight[lk] = torch.zeros_like(edge_aggwight[lk])
            if args.cloudagg == 'sign':
                w_clint = copy.deepcopy(edge_aggwight)
                for lk in edge_aggwight.keys():
                    w_clint[lk] = torch.sign(edge_aggwight[lk] - w_signinit[lk])
                    w_clint[lk] = torch.where(w_clint[lk] == 0, torch.sign(uniform_tensor[lk]), w_clint[lk])
                    sign_aggwight[lk] = sign_aggwight[lk] + w_clint[lk] #-w_signinit[lk]#torch.mean(torch.sign(edge_aggwight[lk].mul(w_signinit[lk]))) *
                    #sign_aggwight[lk] = sign_aggwight[lk] + torch.where(edge_aggwight[lk]-w_signinit[lk] <= 0, negativeone_tensor[lk], ones_tensor[lk])
            if args.cloudagg == 'sign_irls':
                sign_cloudaggwight = copy.deepcopy(edge_aggwight)
                edge_sign = copy.deepcopy(edge_aggwight)
                edgeworngkey = 0
                for lk in edge_aggwight.keys():
                    edge_sign[lk] = torch.sign(edge_aggwight[lk].mul(w_signinit[lk]))
                    edge_sign[lk] = torch.where(edge_sign[lk] == 0, torch.sign(uniform_tensor[lk]), edge_sign[lk])
                    if torch.median(edge_sign[lk]) <= 0:
                        edgeworngkey = edgeworngkey + 1
                if edgeworngkey / len(edge_sign.keys()) > 0.5:
                    directionedge = -1
                else:
                    directionedge = 1
                for lk in edge_aggwight.keys():
                    #sign_cloudaggwight[lk] = torch.where(edge_aggwight[lk]-w_signinit[lk] <= 0, negativeone_tensor[lk], ones_tensor[lk])
                    sign_cloudaggwight[lk] =torch.sign(directionedge * edge_aggwight[lk] - w_signinit[lk]) ## torch.mean(torch.sign(edge_aggwight[lk].mul(w_signinit[lk]))) *
                    sign_cloudaggwight[lk] = torch.where(sign_cloudaggwight[lk] == 0, torch.sign(uniform_tensor[lk]), sign_cloudaggwight[lk])
                edge_cloudweights.append(sign_cloudaggwight)
            edge_weights.append(edge_aggwight)
            loss_avg = sum(loss_locals) / len(loss_locals)
            edge_loss.append(loss_avg)
            print('edge_loss={}'.format(edge_loss))
        # cloud agg
        if args.cloudagg == 'sign':
            total_sign = copy.deepcopy(sign_aggwight)
            for lk in sign_aggwight.keys():
                total_sign[lk] = torch.sign(sign_aggwight[lk])
                total_sign[lk] = torch.where(total_sign[lk] == 0, torch.sign(uniform_tensor[lk]), total_sign[lk])
                #total_sign[lk] = torch.where(sign_aggwight[lk] <= 0, negativeone_tensor[lk], ones_tensor[lk])
        if args.num_edges == 1:
            if args.cloudagg == 'sign':
                for lk in sign_aggwight.keys():
                    w_glob[lk] = w_signinit[lk] + args.gamma2 * total_sign[lk]
            elif args.cloudagg == 'sign_irls':
                w_glob = copy.deepcopy(sign_cloudaggwight)#aggregate_weights_cloud(args, edge_cloudweights, net_glob, reweights, fg)
                w_csiint = copy.deepcopy(sign_cloudaggwight)
                for lk in w_glob.keys():
                    w_csiint[lk] = torch.sign(w_glob[lk])
                    w_csiint[lk] = torch.where(w_csiint[lk] == 0, torch.sign(uniform_tensor[lk]), w_csiint[lk])
                    w_glob[lk] = w_signinit[lk] + args.gamma2 * w_csiint[lk]
                    #w_glob[lk] = w_signinit[lk] + args.gamma * torch.where(w_glob[lk] <= 0, negativeone_tensor[lk], ones_tensor[lk])
            else:
                w_glob = copy.deepcopy(edge_aggwight)
        else:
            if args.cloudagg == 'sign':
                for lk in sign_aggwight.keys():
                    w_glob[lk] = w_signinit[lk] + args.gamma2 * total_sign[lk]
            elif args.cloudagg == 'sign_irls':
                w_glob = aggregate_weights_cloud(args, edge_cloudweights, reweights)
                w_csiint = copy.deepcopy(w_glob)
                for lk in w_glob.keys():
                    w_csiint[lk] = torch.sign(w_glob[lk])
                    w_csiint[lk] = torch.where(w_csiint[lk] == 0, torch.sign(uniform_tensor[lk]), w_csiint[lk])
                    w_glob[lk] = w_signinit[lk] + args.gamma2 * w_csiint[lk]
                    #w_glob[lk] = w_signinit[lk] + args.gamma * torch.where(w_glob[lk] <= 0, negativeone_tensor[lk], ones_tensor[lk])
            else:
                w_glob = aggregate_weights_cloud(args, edge_weights, reweights)
        # copy weight to net_glob
        if not args.agg == 'fg':
            net_glob.load_state_dict(w_glob)

        # test data
        list_acc = test(net_glob, dataset_test, args, test_users)
        accs.append(list_acc)
        test_acc_plot.append(statistics.mean(list_acc))

        # poisoned test data
        if args.is_backdoor:
            _backdoor_acc = backdoor_test(net_glob, dataset_test, args, np.asarray(list(range(len(dataset_test)))))
            backdoor_accs.append([_backdoor_acc])
            last_bd_acc = _backdoor_acc
        # print loss
        train_loss_plot.append(statistics.mean(edge_loss))
        print('Train loss = {}'.format(statistics.mean(edge_loss)))


    save_folder='../save/'

    ######################################################
    # Testing                                            #
    ######################################################
    list_acc, list_loss = [], []
    net_glob.eval()
    if args.dataset == "mnist":
        classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    elif args.dataset == "cifar":
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    elif args.dataset == "fmnist":
        classes = ('t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle-boot')
    elif args.dataset == "loan":
        classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8')

    added_acc = 0
    added_loss = 0
    added_data_num = 0
    with torch.no_grad():
        for c in range(len(classes)):
            if c < len(classes) and len(test_users[c])>0:
                net_local = LocalUpdate(args=args, dataset=dataset_test, idxs=test_users[c], tb=summary, test_flag=True)
                acc, loss = net_local.test(net=net_glob)
                print("test accuracy for label {} is {:.2f}%".format(classes[c], acc * 100.))
                list_acc.append(acc)
                list_loss.append(loss)
                added_acc += acc * len(test_users[c])
                added_loss += loss * len(test_users[c])
                added_data_num += len(test_users[c])
        print("average acc: {:.2f}%,\ttest loss: {:.5f}".format(100. * added_acc / float(added_data_num),
                                                                added_loss / float(added_data_num)))

        if args.is_backdoor:
            acc = backdoor_test(net_glob, dataset_test, args, np.asarray(list(range(len(dataset_test)))))
            print("backdoor success rate for attacker is {:.2f}%".format(acc * 100.))
        else:
            if args.attack_label == -1:
                args.attack_label = 1
            net_local = LocalUpdate(args=args, dataset=dataset_test, idxs=test_users[args.attack_label], tb=summary, attack_label=args.attack_label,
                                    test_flag=True)
            acc, loss = net_local.test(net=net_glob)
            print("attack success rate for attacker is {:.2f}%".format(acc * 100.))

    ######################################################
    # Plot                                               #
    ######################################################
    # plot acc
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    accs_np = np.zeros((len(accs), len(accs[0])))
    for i, w in enumerate(accs):
        accs_np[i] = np.array(w)
    accs_np = accs_np.transpose()
    colormap = plt.cm.gist_ncar  # nipy_spectral, Set1,Paired
    colors = [colormap(i) for i in np.linspace(0, 0.9, len(accs_np))]
    plt.ylabel('accuracy'.format(i))
    for i, y in enumerate(accs_np):
        plt.plot(range(len(y)), y)
    for i, j in enumerate(ax1.lines):
        j.set_color(colors[i])
    plt.legend([str(i) for i in range(len(accs_np))], loc='lower right')
    plt.savefig(
        save_folder+'kindacc_atttype{}_cloudagg{}_gamma{}_user{}_attuser{}_{}_{}_{}_{}_users{}_attackers_{}_attackep_{}_MAX_GRAD_NORM{}_NOISE_MULTIPLIER{}_thresh_{}_iid{}.png'.format(args.attackertype, args.cloudagg, args.gamma1, args.num_users, args.num_attackers, args.agg,args.dataset,
                                                                                           args.model, args.epochs,
                                                                                           args.num_users - args.num_attackers,
                                                                                           args.num_attackers, args.max_grad_norm, args.noise_multiplier,
                                                                                           args.attacker_ep,
                                                                                           args.thresh,args.iid))
    # plot loss curve
    plt.figure()
    plt.plot(range(len(train_loss_plot)), train_loss_plot)
    plt.ylabel('train_loss')
    plt.savefig(save_folder+'loss_atttype{}_cloudagg{}_gamma{}_user{}_attuser{}_{}_{}_{}_{}_C{}_MAX_GRAD_NORM{}_NOISE_MULTIPLIER{}_iid{}.png'.format(args.attackertype, args.cloudagg, args.gamma1, args.num_users, args.num_attackers, args.agg, args.dataset, args.model, args.epochs, args.frac, args.max_grad_norm, args.noise_multiplier, args.iid))

    # plot acc curve
    plt.figure()
    plt.plot(range(len(test_acc_plot)), test_acc_plot)
    plt.ylabel('test_acc')
    plt.savefig(save_folder+'totalacc_atttype{}_cloudagg{}_gamma{}_user{}_attuser{}_{}_{}_{}_{}_C{}_MAX_GRAD_NORM{}_NOISE_MULTIPLIER{}_iid{}.png'.format(args.attackertype, args.cloudagg, args.gamma1, args.num_users, args.num_attackers, args.agg, args.dataset, args.model, args.epochs, args.frac, args.max_grad_norm, args.noise_multiplier, args.iid))

    if args.is_backdoor:
        # plot backdoor acc
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        backdoor_accs_np = np.zeros((len(backdoor_accs), len(backdoor_accs[0])))
        for i, w in enumerate(backdoor_accs):
            backdoor_accs_np[i] = np.array(w)
        backdoor_accs_np = backdoor_accs_np.transpose()
        colormap = plt.cm.gist_ncar  # nipy_spectral, Set1,Paired
        colors = [colormap(i) for i in np.linspace(0, 0.9, len(backdoor_accs_np))]
        plt.ylabel('backdoor success rate'.format(i))
        for i, y in enumerate(backdoor_accs_np):
            plt.plot(range(len(y)), y)
        for i, j in enumerate(ax1.lines):
            j.set_color(colors[i])
        plt.legend([str(i) for i in range(len(backdoor_accs_np))], loc='lower right')
        plt.savefig(
            save_folder+'backdoor_accs_atttype{}_cloudagg{}_gamma{}_user{}_attuser{}_{}_{}_{}_{}_users{}_attackers_{}_attackep_{}_thresh_{}_MAX_GRAD_NORM{}_NOISE_MULTIPLIER{}_iid{}.png'.format(args.attackertype, args.cloudagg, args.gamma1, args.num_users, args.num_attackers, args.agg,args.dataset,
                                                                                                         args.model,
                                                                                                         args.epochs,
                                                                                                         args.num_users - args.num_attackers,
                                                                                                         args.num_attackers,
                                                                                                         args.attacker_ep,args.thresh, args.max_grad_norm, args.noise_multiplier, args.iid))
    if args.is_backdoor:
        WriteToFile(args, backdoor_accs, test_acc_plot, train_loss_plot)
    else:
        backdoor_accs = []
        WriteToFile(args, backdoor_accs, test_acc_plot, train_loss_plot)