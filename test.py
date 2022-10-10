import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from RS_dataset import MiniImageNet
from samplers import CategoriesSampler
from encoder import Convnet_4
from utils import pprint, set_gpu, count_acc, Averager, euclidean_metric,get_logger,cosine_dist,cosine_dist2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    # parser.add_argument('--load', default='./save/proto-5/max-acc.pth')
    parser.add_argument('--batch', type=int, default=600)
    parser.add_argument('--train_way', type=int, default=5)
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=30)
    parser.add_argument('--dataset', type=str, default='NWPU')
    parser.add_argument('--fusionstyle', type=str, default='mfb')
    parser.add_argument('--word2vec', type=str, default='glove840B')
    parser.add_argument('--lambda1', type=float, default=1)
    parser.add_argument('--lambda2', type=float, default=1)
    args = parser.parse_args()
    pprint(vars(args))
    if 'NWPU' in args.dataset:
        dataset_name = 'NWPU-RESISC45'
    elif 'RSD46' in args.dataset:
        dataset_name = 'RSD46-WHU'
    elif 'RSSDIVCS' in args.dataset:
        dataset_name = 'RSSDIVCS'
    elif 'WHU' in args.dataset:
        dataset_name = 'WHU-RS19'
    elif 'UCM' in args.dataset:
        dataset_name = 'UCMerced_LandUse'
    elif 'RSSDIVCS' in args.dataset:
        dataset_name = 'RSSDIVCS'
    elif 'AID' in args.dataset:
        dataset_name = 'AID'
    elif 'million' in args.dataset:
        dataset_name = 'million-AID'
    elif 'OPTIMAL' in args.dataset:
        dataset_name = 'OPTIMAL-31'
    elif 'PatternNet' in args.dataset:
        dataset_name = 'PatternNet'
    else:
        raise (ValueError, 'Unsupported dataset')
    if 'bert' in args.word2vec:
        word2vec_length = 1024
    elif args.word2vec == 'glove840B':
        word2vec_length = 300
    elif args.word2vec == 'fasttext':
        word2vec_length = 300
    elif args.word2vec == 'KG':
        word2vec_length = 50
    else:
        raise (ValueError, 'Unsupported word2vec')


    dataset = MiniImageNet('test',dataname=dataset_name,word_embedding=args.word2vec)
    sampler = CategoriesSampler(dataset.label,
                                args.batch, args.way, args.shot + args.query)
    loader = DataLoader(dataset, batch_sampler=sampler,
                        num_workers=20, pin_memory=True)


    # load_path = './save/' + dataset_name + '_proto' + str(args.shot) + 'shot'+ args.word2vec + args.fusionstyle + '_'+str(args.lambda1)+'_'+str(args.lambda2) +'/max-acc.pth'
    # log_file = './save/' + dataset_name + '_proto' + str(args.shot) + 'shot' + args.word2vec + args.fusionstyle + '_'+str(args.lambda1)+'_'+str(args.lambda2) + '/exp_acc_test.log'
    load_path = './save/' + args.dataset + '_proto_' + str(args.train_way) + 'trainway_' + str(args.train_way) + 'testway_' + str(args.shot) + 'shot_'+ args.word2vec + '_'+ args.fusionstyle + '_'+str(int(args.lambda1))+'_'+str(int(args.lambda2))+'/epoch-last.pth' #max-acc.pth
    log_file = './save/' + args.dataset + '_proto_' + str(args.train_way) + 'trainway_' + str(args.train_way) + 'testway_' + str(args.shot) + 'shot_' + args.word2vec + '_'+ args.fusionstyle + '_'+str(int(args.lambda1))+'_'+str(int(args.lambda2)) + '/exp_acc_test.log'

    set_gpu(args.gpu)
    model = Convnet_4(shot=args.shot,word2vec_length=word2vec_length,fusion=args.fusionstyle).cuda()
    print(load_path)
    model.load_state_dict(torch.load(load_path))
    model.eval()
    acc_all = []

    logger = get_logger(log_file)
    logger.info(vars(args))
    logger.info('start test!!!')
    ave_acc = Averager()
    import time

    time_start = time.time()

    for i, batch in enumerate(loader, 1):
        data, _,_,label_real = [_.cuda() for _ in batch]
        k = args.way * args.shot
        data_shot, data_query = data[:k], data[k:]
        classname_support = label_real[:k]  # 5*300

        text_proto,vis_proto = model.forward(data_shot,classname_support)

        query_feature = model.forward_query(data_query)
        vis_dist = cosine_dist2(query_feature, vis_proto,scale_weight=50)  # (450,30)  30指的是train_way,15*30每个类别１５个query样本
        text_dist = cosine_dist2(query_feature, text_proto,scale_weight=20)
        # print(vis_dist[0],text_dist[0])
        # print(vis_dist,text_dist)

        logits = (args.lambda1* vis_dist + args.lambda2*text_dist)
        # logits = vis_dist

        label = torch.arange(args.way).repeat(args.query)
        label = label.type(torch.cuda.LongTensor)
        acc = count_acc(logits, label)
        ave_acc.add(acc)
        logger.info('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))

        acc_all.append(acc * 100)

        x = None; p = None; logits = None
    time_end = time.time()
    print(time_start, time_end, 'totally cost:', (time_end - time_start) * 1000 / 600, "ms")  # inference time


    result_txt_path = log_file.replace('.log','.txt')
    acc_all_array = np.asarray(acc_all)
    acc_mean = np.mean(acc_all_array)
    acc_std = np.std(acc_all_array)
    logger.info('batch_num {}: {:.2f}(+_{:.2f})'.format(args.batch, acc_mean, 1.96*acc_std/np.sqrt(args.batch)))
    with open(result_txt_path, 'a') as f:
        f.write('batch_num :{},mean {:.2f}(std:+_{:.2f})'.format(args.batch, acc_mean, 1.96*acc_std/np.sqrt(args.batch)))



