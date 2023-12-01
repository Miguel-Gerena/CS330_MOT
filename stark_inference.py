import torch
import sys

# Add the directory to the sys path
sys.path.append('/home/LavinuxCS330azureuser/CS330_MOT_Project_V2/CS330_MOT/Stark')

model = torch.load("/home/LavinuxCS330azureuser/CS330_MOT_Project_V2/CS330_MOT/Stark/checkpoints/train/stark_st2/baseline_R101/STARKST_ep0050.pth.tar")
model2 = torch.load_state_dict("/home/LavinuxCS330azureuser/CS330_MOT_Project_V2/CS330_MOT/Stark/checkpoints/train/stark_st2/baseline_R101/STARKST_ep0050.pth.tar")

print(model)





# python tracking/test_lavi.py --tracker_name stark_st --tracker_param baseline_R101 --dataset coco --threads 32
# python tracking/analysis_results.py # need to modify tracker configs and names