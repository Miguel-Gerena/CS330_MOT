import shutil
import subprocess
import os
import sys

sys.path.append("./Deep-EIoU/Deep-EIoU/")
from ak_make_SCT_folder import copy_results


def process_results(experiment_name, base_path = "D:/classes/CS330/project/CS330_MOT"):
    os.chdir(f"{base_path}/Deep-EIoU/Deep-EIoU")
    base = f"{base_path}/Deep-EIoU/Deep-EIoU/YOLOX_outputs/{experiment_name}"
    dest = f"{base_path}/Deep-EIoU/SCT/{experiment_name}"
    copy_results(base, dest)
    subprocess.run(["python", "tools/sport_interpolation.py", "--experiment-name", f"{experiment_name}", "--root_path", f"{base_path}/Deep-EIoU/Deep-EIoU", "--txt_path", f"{base_path}/Deep-EIoU/Deep-EIoU" ])

    # TODO manual setup - make sure the test file has the correct videos and those are located in the test folder as well
    source_dir = f'{base_path}/Deep-EIoU/interpolation/{experiment_name}'
    destination_dir = f'{base_path}/TrackEval/data/res'

    # Check if the destination directory exists, if not, create it
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    # Copy the directory
    shutil.copytree(source_dir, os.path.join(destination_dir, experiment_name), dirs_exist_ok=True)

    os.chdir(f'{base_path}/TrackEval')
    subprocess.run(["python","./scripts/run_mot_challenge.py","--BENCHMARK","sportsmot","--SPLIT_TO_EVAL","test","--METRICS","HOTA","CLEAR","Identity","VACE","--USE_PARALLEL","False","--PRINT_CONFIG","True","--GT_FOLDER","./data/ref","--TRACKERS_FOLDER","./data/res","--OUTPUT_FOLDER","./output/","--SEQMAP_FILE","./data/ref/seqmaps/sportsmot-test.txt"])


# process_results("23.support_6.query_3.inner_steps_1.inner_lr_0.006.learn_inner_lrs_True.outer_lr_0.001.batch_size_5.train_iter_2000000..val_iter_3")