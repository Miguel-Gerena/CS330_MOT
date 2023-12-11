import shutil
import subprocess
import os
import sys
import pandas as pd

sys.path.append("./Deep-EIoU/Deep-EIoU/")
from ak_make_SCT_folder import copy_results

base_path = "D:/classes/CS330/project/CS330_MOT"
deep = f"{base_path}/Deep-EIoU/Deep-EIoU/"
def process_results(experiment_name, base_path = "D:/classes/CS330/project/CS330_MOT"):
    os.chdir(f"{base_path}/Deep-EIoU/Deep-EIoU")
    base = f"{base_path}/Deep-EIoU/Deep-EIoU/YOLOX_outputs/{experiment_name}"
    dest = f"{base_path}/Deep-EIoU/SCT/{experiment_name}"
    copy_results(base, dest)
    subprocess.run(["python", "tools/sport_interpolation.py", "--experiment-name", f"{experiment_name}", "--root_path", f"{base_path}/Deep-EIoU/", "--txt_path", f"{base_path}/Deep-EIoU/" ])

    # TODO manual setup - make sure the test file has the correct videos and those are located in the test folder as well
    source_dir = f'{base_path}/Deep-EIoU/interpolation/{experiment_name}'
    destination_dir = f'{base_path}/TrackEval/data/res'

    # Check if the destination directory exists, if not, create it
    if not os.path.exists(destination_dir ):
        os.makedirs(destination_dir)
    # Copy the directory
    shutil.copytree(source_dir, os.path.join(destination_dir, experiment_name), dirs_exist_ok=True)

    os.chdir(f'{base_path}/TrackEval')
    subprocess.run(["python","./scripts/run_mot_challenge.py","--BENCHMARK","sportsmot","--SPLIT_TO_EVAL","test","--METRICS","HOTA","CLEAR","Identity","VACE","--USE_PARALLEL","False","--PRINT_CONFIG","True","--GT_FOLDER","./data/ref","--TRACKERS_FOLDER","./data/res","--OUTPUT_FOLDER","./output/","--SEQMAP_FILE","./data/ref/seqmaps/sportsmot-test.txt"])
    # shutil.rmtree(destination_dir + "/" + experiment_name)

def move_files():
    src =  "YOLOX_outputs/23.support_6.query_3.inner_steps_1.inner_lr_0.006.learn_inner_lrs_True.outer_lr_0.001.batch_size_5.train_iter_2000000..val_iter_3"
    des = "./result/"
    folders = os.listdir(src)
    for folder in folders:
        if folder == "v_-9kabh1K8UA_c008":
            continue
        os.rename(f"{deep}{src}/{folder}/{folder}_results.txt", f"{deep}{des}{folder}_results.txt")

def process_output_id():
    for file in os.listdir(f"{deep}/result"):
        if file[-3:] == "txt":
            data = pd.read_csv(f"{deep}/result/{file}", sep=",", header=None)
            data[6] = -1
            data[1] = data[1] - 1
            data.to_csv(f"{deep}/result_processed/{file[:-12]}.txt", sep=',', index=False, header=False)

def process_interpolation():
    deep = "D:/classes/CS330/project/CS330_MOT/Deep-EIoU/Deep-EIoU/YOLOX_outputs"
    for folders in os.listdir(f"{deep}"):
        for folder in os.listdir(f"{deep}/{folders}"):
            for file in os.listdir(f"{deep}/{folders}/{folder}"):
                if file[-3:] == "txt":
                    data = pd.read_csv(f"{deep}/{folders}/{folder}/{file}", sep=",", header=None)
                    # data[6] = -1
                    data[1] = data[1] - 1
                    data.to_csv(f"{deep}/{folders}/{folder}/{file}", sep=',', index=False, header=False)
