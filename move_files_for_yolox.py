

from genericpath import exists
import os
import shutil
base_dest = "./data_cs/combined_train_val"
base_source = "./TrackEval/data/ref/sportsmot-test/"
folders = set(os.listdir(base_source))

for folder in os.listdir("D:/classes/CS330/project/CS330_MOT/data_cs/sportsmot_publish/dataset/val"):
    if folder in folders:
        os.rename(f"{base_source}/{folder}", f"{base_dest}{folder}")

