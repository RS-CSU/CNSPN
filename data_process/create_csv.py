# -*- ecoding: utf-8 -*-
# @Time: 7/9/21 10:07 AM
# @Author: guoya
# @email: 1203392419@qq.com
# @ModuleName: NWPU-45
# @Describe: NWPU45创建train.csv val.csv test.csv。首行filename,label;第一列文件名，第二列标签。
import os
import csv

def generate_csv(dataset_dir = '/home/zhuofeng/dataset/小样本数据集/NWPU-RESISC45'):
    data_choice = ['images_background', 'images_evaluation', 'images_test']
    save_choice = ['train.csv', 'val.csv', 'test.csv']
    for k in range(len(data_choice)):
        item = data_choice[k]
        save_path = os.path.join(dataset_dir,save_choice[k])
        with open(save_path,'w') as f:
            f.write("filename,label\n")
            data_path_item = os.path.join(dataset_dir,item) #获取训练集路径
            class_name_list = os.listdir(data_path_item)#获取训练集中的类名
            for class_name in class_name_list:
                image_file_path = os.path.join(data_path_item,class_name)
                file = os.listdir(image_file_path)
                for m,i in enumerate(file):
                    # if m < 500:
                    content = os.path.join(image_file_path,i) +','+ class_name +'\n'
                    f.writelines(content)

if __name__ == '__main__':
    # NWPU_dataset_dir = '/media/gy/study/paper_experment/01paper/datasets/NWPU-RESISC45'
    # generate_csv(dataset_dir=NWPU_dataset_dir)
    #
    # UCMerced_dir = '/media/gy/study/paper_experment/01paper/datasets//UCMerced_LandUse'
    # WHU_dir = '/media/gy/study/paper_experment/01paper/datasets//WHU-RS19'
    # generate_csv(dataset_dir=UCMerced_dir)
    # generate_csv(dataset_dir=WHU_dir)


    # RSD46_WHU_dataset_dir = '/media/gy/study/paper_experment/01paper/datasets/RSD46-WHU'
    # generate_csv(dataset_dir=RSD46_WHU_dataset_dir)

    # RSD46_WHU_dataset_dir = '/media/gy/study/paper_experment/01paper/datasets/RSSDIVCS'
    # generate_csv(dataset_dir=RSD46_WHU_dataset_dir)


    AID_dir = '/media/gy/study/paper_experment/01paper/datasets//AID'
    # millon_AID_dir = '/media/gy/study/paper_experment/01paper/datasets//million-AID'
    # patterent_dir = '/media/gy/study/paper_experment/01paper/datasets//PatternNet'
    # optimal_dir = '/media/gy/study/paper_experment/01paper/datasets//OPTIMAL-31'
    generate_csv(dataset_dir=AID_dir)
    # generate_csv(dataset_dir=millon_AID_dir)
    # generate_csv(dataset_dir=patterent_dir)
    # generate_csv(dataset_dir=optimal_dir)