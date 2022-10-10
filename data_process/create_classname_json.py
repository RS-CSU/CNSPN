# -*- ecoding: utf-8 -*-
# @Time: 7/9/21 11:45 AM
# @Author: guoya
# @email: 1203392419@qq.com
# @ModuleName: create_classname_json
# @Describe: 创建类名和类编号的字典。
import json
import os.path as osp
def creat_classname_json(dataset_path,output_path):
    dataset_dir = dataset_path
    classname_dict = {}
    wnids = []
    setnames = ['train','val','test']
    for setname in setnames:
        train_csv_path = osp.join(dataset_dir, setname + '.csv')
        lines = [x.strip() for x in open(train_csv_path, 'r').readlines()][1:]
        for l in lines:
            name, wnid = l.split(',')
            if wnid not in wnids:
                wnids.append(wnid)
    wnids.sort()
    for index in range(len(wnids)):
        classname_dict[wnids[index]] = index
    with open(output_path,'w') as file:
        json.dump(classname_dict,file)
        file.close()
if __name__ =='__main__':
    # dataset_dir = '/home/zhuofeng/dataset/小样本数据集/NWPU-RESISC45'
    # output_dir = '/home/zhuofeng/dataset/小样本数据集/NWPU-RESISC45/classname_dict.json'
    # creat_classname_json(dataset_dir,output_dir)
    #
    # dataset_dir = '/home/zhuofeng/dataset/小样本数据集/UCMerced_LandUse'
    # output_dir = '/home/zhuofeng/dataset/小样本数据集/UCMerced_LandUse/classname_dict.json'
    # creat_classname_json(dataset_dir,output_dir)
    #
    # dataset_dir = '/home/zhuofeng/dataset/小样本数据集/WHU-RS19'
    # output_dir = '/home/zhuofeng/dataset/小样本数据集/WHU-RS19/classname_dict.json'
    # creat_classname_json(dataset_dir, output_dir)

    # dataset_dir = '/media/gy/study/paper_experment/01paper/datasets/RSD46-WHU'
    # output_dir = '/media/gy/study/paper_experment/01paper/datasets/RSD46-WHU/classname_dict.json'
    # creat_classname_json(dataset_dir, output_dir)

    # dataset_dir = '/media/gy/study/paper_experment/01paper/datasets/RSSDIVCS'
    # output_dir = '/media/gy/study/paper_experment/01paper/datasets/RSSDIVCS/classname_dict.json'
    # creat_classname_json(dataset_dir, output_dir)

    dataset_dir = '/media/gy/study/paper_experment/01paper/datasets/AID'
    output_dir = '/media/gy/study/paper_experment/01paper/datasets/AID/classname_dict.json'
    creat_classname_json(dataset_dir, output_dir)
    dataset_dir = '/media/gy/study/paper_experment/01paper/datasets/million-AID'
    output_dir = '/media/gy/study/paper_experment/01paper/datasets/million-AID/classname_dict.json'
    creat_classname_json(dataset_dir, output_dir)
    dataset_dir = '/media/gy/study/paper_experment/01paper/datasets/PatternNet'
    output_dir = '/media/gy/study/paper_experment/01paper/datasets/PatternNet/classname_dict.json'
    creat_classname_json(dataset_dir, output_dir)
    dataset_dir = '/media/gy/study/paper_experment/01paper/datasets/OPTIMAL-31'
    output_dir = '/media/gy/study/paper_experment/01paper/datasets/OPTIMAL-31/classname_dict.json'
    creat_classname_json(dataset_dir, output_dir)
    dataset_dir = '/media/gy/study/paper_experment/01paper/datasets/UCMerced_LandUse'
    output_dir = '/media/gy/study/paper_experment/01paper/datasets/UCMerced_LandUse/classname_dict.json'
    creat_classname_json(dataset_dir, output_dir)