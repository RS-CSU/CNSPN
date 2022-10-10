import os
import shutil
import time
import pprint
import torch.nn.functional as F
import torch


def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def ensure_path(path):
    if os.path.exists(path):
        if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()


def dot_metric(a, b):
    return torch.mm(a, b.t())


def euclidean_metric(a, b):
    '''
    :param a: query 的嵌入，450*1600
    :param b: support的嵌入　30*1600
    :return: 450*30维度的logits
    '''
    n = a.shape[0]                          #450
    m = b.shape[0]                          #30
    assert a.size(1) == b.size(1)
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2) #注意！！！！这里已经添加负号了
    return logits
def cosine_dist(a,b):
    #a: N x D query
    #b: M x D  proto
    a = F.normalize(a, p=2, dim=1, eps=1e-12)
    b = F.normalize(b, p=2, dim=1, eps=1e-12)
    n = a.shape[0]
    m = b.shape[0]
    d = a.size(1)
    assert a.size(1) == b.size(1)

    bT = torch.transpose(b,1,0)
    output = a @ bT
    return output

def cosine_dist2(a,b,dist='cos',scale_weight=50):
    #a: N x D query
    #b: M x D  proto
    n = a.shape[0]  # 450
    m = b.shape[0]  # 30
    assert a.size(1) == b.size(1)
    embed = a.unsqueeze(1).expand(n, m, -1)
    proto = b.unsqueeze(0).expand(n, m, -1)
    if dist == "l2":
        scores = -torch.pow(embed - proto,2).sum(-1)
    if dist == 'cos':
        scores = (F.cosine_similarity(embed,proto,dim=-1,eps=1e-30)+1 )/2 *scale_weight
    return scores

class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)


def l2_loss(pred, label):
    return ((pred - label)**2).sum() / len(pred) / 2
import pickle
def save_dict(file,dict):
    """
    :param file: xxxx.pkl
    :param dict:
    :return:
    """
    with open(file,'wb') as f:
        pickle.dump(dict,f,pickle.HIGHEST_PROTOCOL)
def load_dict(file):
    with open(file,'rb') as f:
        return pickle.load(f)


import logging
def get_logger(filename, verbosity=1, name=None):
    # 作者：Victor
    # 链接：https: // www.zhihu.com / question / 361602016 / answer / 942037512
    # 来源：知乎
    # 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

