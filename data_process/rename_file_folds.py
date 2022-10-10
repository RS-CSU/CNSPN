import os,sys
def rename_subfolders(path=None):
    old_names = os.listdir(path)
    for old_name in old_names:
        fold2 = os.listdir(os.path.join(path,old_name))
        fold2.sort()
        new_name = fold2[0][:-5]
        print(old_name,fold2[0],new_name)

        os.rename(os.path.join(path,old_name), os.path.join(path,new_name))


path = "/media/gy/study/paper_experment/01paper/datasets/RSSDIVCS/images_background"
rename_subfolders(path)

path = "/media/gy/study/paper_experment/01paper/datasets/RSSDIVCS/images_evaluation"
rename_subfolders(path)

path = "/media/gy/study/paper_experment/01paper/datasets/RSSDIVCS/images_test"
rename_subfolders(path)