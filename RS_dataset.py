# -*- ecoding: utf-8 -*-
# @Time: 7/10/21 10:32 AM
# @Author: guoya
# @email: 1203392419@qq.com
# @ModuleName: RS_dataset
# @Describe: RS的datase,用于语义信息辅助的小样本学习
import os.path as osp
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import json
from utils import load_dict
class MiniImageNet(Dataset):
    def __init__(self, setname, dataname='NWPU',word_embedding='glove840B'):
        csv_path = osp.join('data', dataname +'/'+ setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
        classname_dict_path = './data/' + dataname + '/classname_dict.json'
        with open(classname_dict_path, 'r') as file:
            classname_dict = json.load(file)
            file.close()
        self.classname_dict = classname_dict  #类名字典
        word_embedding_path = './data/' + dataname + '/classname_word2vec_'+word_embedding +'.json'

        with open(word_embedding_path, 'r') as file:#词向量字典
            word2vec = json.load(file)
            file.close()
        self.word2vec = word2vec
        self.setname = setname

        data = [] #
        label = []#存储假标签。每个任务的标签：如５way[0,1,2,3,4]
        label_real = []#存储真标签０－４４
        word_embed = []#存储词向量
        self.class_names = []#存储类别（str）
        lb = -1
        # print(self.word2vec.keys())
        for l in lines:
            path_name, class_name = l.split(',')
            # path = osp.join(ROOT_PATH, 'images', name)
            path = path_name
            # class_name = class_name.lower()
            if class_name not in self.class_names:
                self.class_names.append(class_name)
                lb += 1
            data.append(path)
            label.append(lb)
            label_real.append(self.classname_dict[class_name])  # 标签０－4４
            word_embed.append(self.word2vec[class_name.lower()])#词向量

        self.data = data
        self.label = label
        self.label_real = label_real
        self.word_embed = word_embed

        self.transform = transforms.Compose([
            transforms.Resize([256,256]),
            # transforms.CenterCrop(224),
            transforms.RandomCrop(224),
            transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ])
        self.transform2 = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label, label_real, label_embed= self.data[i], self.label[i],self.label_real[i],self.word_embed[i]
        if self.setname == 'train':
            image = self.transform(Image.open(path).convert('RGB'))
        else:
            image = self.transform2(Image.open(path).convert('RGB'))
        label_embed = torch.from_numpy(np.array(label_embed)).float()
        # word_embed = np.array(self.word2vec[str(label_real)])
        # # print(image.shape,word_embed.shape)
        # word_embed = torch.from_numpy(word_embed).float() #直接加载是double，需要转成float
        # # print(image.shape, word_embed.shape)
        return image, label, label_real,label_embed #返回图像，标签（5way:0-4），标签真值(0-44),标签嵌入
