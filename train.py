import argparse
import os.path as osp
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# from mini_imagenet import MiniImageNet
from RS_dataset import MiniImageNet
from samplers import CategoriesSampler
from encoder import Convnet_4

from utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric,get_logger,cosine_dist,cosine_dist2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--save-epoch', type=int, default=10)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=5)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--dataset', type=str, default='NWPU')
    parser.add_argument('--word2vec', type=str, default='glove840B')
    parser.add_argument('--fusionstyle', type=str, default='mfb')
    parser.add_argument('--lambda1', type=float, default=1)
    parser.add_argument('--lambda2', type=float, default=1)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    if 'NWPU' in args.dataset: ##'AID' 'WHU' 'UCM' 'OPTIMAL'  'PatternNet'
        dataset_name = 'NWPU-RESISC45'
        max_epoch = 150
        lr_init = 1e-3
    elif 'RSD46' in args.dataset:
        dataset_name = 'RSD46-WHU'
        max_epoch = 300
        lr_init = 1e-3
    if 'bert' in args.word2vec:
        word2vec_length = 1024
    elif args.word2vec == 'glove840B':
        word2vec_length = 300
    elif args.word2vec == 'fasttext':
        word2vec_length = 300
    else:
        raise (ValueError, 'Unsupported word2vec')


    trainset = MiniImageNet('train',dataname=dataset_name,word_embedding=args.word2vec)  #64类,标签0-63，38400个样本

    train_sampler = CategoriesSampler(trainset.label, 100,
                                      args.train_way, args.shot + args.query) #label, n_batch, n_cls, n_per
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                              num_workers=20, pin_memory=True)
    valset = MiniImageNet('val',dataname=dataset_name,word_embedding=args.word2vec)     #16类,标签0-15，9600个样本
    #测试集：20类,标签0-19，12000个样本

    val_sampler = CategoriesSampler(valset.label, 100,
                                    args.test_way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=20, pin_memory=True)


    model = Convnet_4(shot=args.shot,word2vec_length=word2vec_length,fusion=args.fusionstyle).cuda()

    save_path = './save/' + args.dataset + '_proto_' + str(args.train_way) + 'trainway_' + str(args.test_way) + 'testway_' + str(args.shot) + 'shot_'+ args.word2vec + '_'+ args.fusionstyle + '_'+str(int(args.lambda1))+'_'+str(int(args.lambda2))
    log_file = './save/' + args.dataset + '_proto_' + str(args.train_way) + 'trainway_' + str(args.test_way) + 'testway_' + str(args.shot) + 'shot_' + args.word2vec + '_'+ args.fusionstyle + '_'+str(int(args.lambda1))+'_'+str(int(args.lambda2)) +'/exp.log'
    set_gpu(args.gpu)
    ensure_path(save_path)


    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init,weight_decay=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)#降低到1e-4


    def save_model(name):
        torch.save(model.state_dict(), osp.join(save_path, name + '.pth'))


    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0

    # trlog['val_vis_loss'] = []
    # trlog['val_text_loss'] = []

    timer = Timer()
    logger = get_logger(log_file)
    logger.info(vars(args))
    logger.info('start training!!!')
    # weight_tensor =torch.FloatTensor([9.0])
    # weight_var = Variable(weight_tensor,requires_grad=True).cuda()

    for epoch in range(1, max_epoch + 1):

        model.train()
        tl = Averager()
        ta = Averager()
        for i, batch in enumerate(train_loader, 1):
            data, label_1, label_rl1 ,label_real = [_.cuda() for _ in batch]

            p = args.shot * args.train_way #0-450-480
            data_shot, data_query = data[:p], data[p:]  #([30, 3, 84, 84]),([450, 3, 84, 84])
            classname_support = label_real[:p] #30*300

            text_proto,vis_proto = model(data_shot,classname_support) #torch.Size([5, 512])


            label = torch.arange(args.train_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor) #torch.Size([450])

            query_feature = model.forward_query(data_query)
            vis_dist = cosine_dist2(query_feature, vis_proto)   #(450,30)  30指的是train_way,15*30每个类别１５个query样本
            text_dist = cosine_dist2(query_feature,text_proto,scale_weight=10)
            logits = (args.lambda1 * vis_dist + args.lambda2 * text_dist)#/(args.lambda1 + args.lambda2)


            loss = torch.nn.CrossEntropyLoss()(vis_dist, label) + torch.nn.CrossEntropyLoss()(text_dist, label)
            acc = count_acc(logits, label)
            if (i+1)%33 == 0:
                logger.info('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'.format(epoch, i, len(train_loader), loss.item(), acc))

            tl.add(loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            proto = None;
            logits = None;
            loss = None
        lr_scheduler.step()


        tl = tl.item()
        ta = ta.item()

        model.eval()#如果训练和测试的way不一致，这里需要重新加载当前模型
        vl = Averager()
        va = Averager()

        for i, batch in enumerate(val_loader, 1):
            data, _, _,label_real = [_.cuda() for _ in batch]
            p = args.shot * args.test_way
            data_shot, data_query  = data[:p], data[p:] #[5, 3, 84, 84]),([75, 3, 84, 84])
            classname_support = label_real[:p]  # 5*300

            text_proto,vis_proto = model(data_shot,classname_support)
            # proto = proto.reshape(args.shot, args.test_way, -1).mean(dim=0)


            label = torch.arange(args.test_way).repeat(args.query) #0-30，重复１５次
            label = label.type(torch.cuda.LongTensor)

            query_feature = model.forward_query(data_query)
            vis_dist = cosine_dist2(query_feature, vis_proto,scale_weight=50)  # (450,30)  30指的是train_way,15*30每个类别１５个query样本
            text_dist = cosine_dist2(query_feature, text_proto,scale_weight=10)
            logits = (args.lambda1 * vis_dist + args.lambda2 * text_dist)  # /(args.lambda1 + args.lambda2)

            loss1 = torch.nn.CrossEntropyLoss()(vis_dist, label)
            loss2 = torch.nn.CrossEntropyLoss()(text_dist, label)
            loss = loss1 +loss2
            acc = count_acc(logits, label)
            vl.add(loss.item())

            va.add(acc)

            proto = None;
            logits = None;
            loss = None


        vl = vl.item()
        # vls = vls.item()#视觉
        # vlt = vlt.item()#文本
        va = va.item()
        logger.info('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

        # print(weight_var.item())

        logger.info('epoch {}, val, loss={:.4f},text_loss={:.4f}，vis_loss={:.4f}， acc={:.4f}'.format(epoch, vl, loss2, loss1, va))

        if va > trlog['max_acc']:
            trlog['max_acc'] = va
            save_model('max-acc')

        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)

        torch.save(trlog, osp.join(save_path, 'trlog'))

        save_model('epoch-last')

        # if epoch % args.save_epoch == 0:
        #     save_model('epoch-{}'.format(epoch))

        logger.info('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / max_epoch)))
    logger.info("finish training")
    logger.info("best val acc:{}".format(trlog['max_acc']))