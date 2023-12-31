import argparse
import re
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# import tensorflow as tf
import os.path as osp
import numpy as np
import time
import math
import cv2
import torch
import gc
import random
import util
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch import nn
import torch.nn.functional as F
from torch import autograd
from torcheval.metrics import MulticlassF1Score, MulticlassAccuracy
from torcheval.metrics.functional import multiclass_accuracy
import sys
sys.path.append('.')
# sys.path.append('Deep-EIoU/Deep-EIoU/')


from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracking_utils.timer import Timer

from tracker.Deep_EIoU import Deep_EIoU
from reid.torchreid.utils import FeatureExtractor
import torchvision.transforms as T
from torchreid.utils import load_pretrained_weights

from pathlib import Path
from torch.utils import tensorboard
from tqdm import tqdm
if os.getlogin() == "darke":
    base_path ="D:/classes/CS330/project/CS330_MOT"
else:
    base_path = 'C:/Users/akayl/Desktop/CS330_MOT'
sys.path.append(base_path)
from data_cs import DataGenerator, process_results
from torch.utils.data import DataLoader, BatchSampler, RandomSampler
from torchreid.utils import (check_isfile, load_pretrained_weights, compute_model_complexity)
from sklearn.metrics import log_loss
from torchreid.models import build_model
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

NUM_INPUT_CHANNELS = 1 # num batch
SUMMARY_INTERVAL = 10
SAVE_INTERVAL = 100
LOG_INTERVAL = 10
VAL_INTERVAL = 10
NUM_TEST_TASKS = 600
NUM_CLASSES_FRAME_ID = 23  # Number of classes for frame ID
NUM_CLASSES_PLAYER_ID = 23  # Number of classes for player ID
WEIGHT_BBOX = 4.0
WEIGHT_PLAYER_ID = 1.0
ONE_ZERO_LABELS = False



def make_parser():
    parser = argparse.ArgumentParser("DeepEIoU Demo")
    parser.add_argument("-expn", "--experiment-name", type=str, default="sportsmot-test", help="should be: sportsmot-dataset_type (IE: test, train, val)")
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
"--path", default=base_path + "/data_cs/sportsmot_publish/dataset/test" if os.getlogin() == "darke" else "/dataset/test", help="path to images or video"
    )
    parser.add_argument(
        "--save_result",
        default=True,
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default="yolox/yolox_x_ch_sportsmot.py",
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=25, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold valid for tracks")
    parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=60, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--nms_thres", type=float, default=0.7, help='nms threshold')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

    # reid args
    parser.add_argument("--with-reid", dest="with_reid", default=True, action="store_true", help="use Re-ID flag.")
    parser.add_argument('--proximity_thresh', type=float, default=0.5, help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25, help='threshold for rejecting low appearance similarity reid matches')

    # maml args
    parser.add_argument('--fine_tune', default=False, action='store_true', help='uses pretrained weights and fine tunes')
    parser.add_argument('--log_dir', type=str, default=None, help='directory to save to or load from')
    parser.add_argument('--model', type=str, default='maml', help='model to run')
    parser.add_argument('--num_videos', type=int, default=1, help='number of videos to include in the support set')
    parser.add_argument('--num_way', type=int, default=23, help='number of classes in a task')
    parser.add_argument('--num_sports', type=int, default=3, help='number of sports')
    parser.add_argument('--num_support', type=int, default=25, help='number of support examples per class in a task')
    parser.add_argument('--num_query', type=int, default=3, help='number of query examples per class in a task')
    parser.add_argument('--meta_batch_size', type=int, default=5, help='number of tasks per outer-loop update')
    parser.add_argument('--meta_batch_size_miguel', type=int, default=10, help='number of larger batch size when low on memory')

    parser.add_argument('--test_batch_size', type=int, default=6, help='number of tasks per outer-loop update')
    parser.add_argument('--meta_train_iterations', type=int, default=2000000, help='number of baches of tasks to iterate through for train')
    parser.add_argument('--meta_val_iterations', type=int, default=3, help='number of baches of tasks to iterate through for val per every check')
    parser.add_argument('--num_inner_steps', type=int, default=1, help='number of inner-loop updates')
    parser.add_argument('--inner_lr', type=float, default=0.006, help='inner-loop learning rate initialization')
    parser.add_argument('--learn_inner_lrs', default=True, action='store_true', help='whether to optimize inner-loop learning rates')
    parser.add_argument('--outer_lr', type=float, default=0.001, help='outer-loop learning rate')
    parser.add_argument('--test', default=False, action='store_true', help='train or test')
    parser.add_argument('--num_workers', type=int, default=4, help=('needed to specify dataloader'))
    parser.add_argument('--checkpoint_step', type=int, default=-1, help=('checkpoint iteration to load for resuming training, or for evaluation (-1 is ignored)'))
    parser.add_argument('--cache', action='store_true')

    parser.add_argument('--eval', default=False, action='store_true', help='wether to evaluate the feature extractor model or not')
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    image_names.sort()
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
        return outputs, img_info
    
def safe_crop(frame, x1, y1, x2, y2):
    # Ensure the coordinates are within the image boundaries
    h, w = frame.shape[:2]
    x1, x2 = max(0, int(x1)), min(int(x2), w)
    y1, y2 = max(0, int(y1)), min(int(y2), h)
    # Ensure width and height are positive
    if x2 > x1 and y2 > y1:
        return frame[y1:y2, x1:x2]
    else:
        return None


def process_sequence(predictor, extractor, img_dir, vis_folder, args):
    image_list = get_image_list(img_dir)
    if not image_list:
        logger.error("No images found in the specified folder.")
        return
    
    video_id = os.path.basename(os.path.dirname(img_dir))

    # Use the first image to get width and height
    first_frame = cv2.imread(image_list[0])
    height, width = first_frame.shape[:2]


    save_folder = osp.join(vis_folder, video_id)
    os.makedirs(save_folder, exist_ok=True)

    # Use video_id in the save_path for clarity
    save_path = osp.join(save_folder, video_id + ".mp4")
    logger.info(f"video save_path is {save_path}")


    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height)
    )

    tracker = Deep_EIoU(args, frame_rate=args.fps)
    timer = Timer()
    frame_id = 0
    results = []

    for img_path in image_list:
        try:
            frame = cv2.imread(img_path)
            if frame is None:
                logger.warning(f"Failed to read image {img_path}")
                continue

            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id+1, args.fps))
            
            outputs, img_info = predictor.inference(frame, timer)
            det = outputs[0].cpu().detach().numpy() if outputs[0] is not None else None

            if det is not None:
                scale = min(1440/width, 800/height)
                det /= scale
                
                valid_detections = []
                cropped_imgs = []
                for detection in det:
                    x1, y1, x2, y2, _, _, _ = detection
                    crop = safe_crop(frame, x1, y1, x2, y2)
                    if crop is not None:
                        cropped_imgs.append(crop)
                        valid_detections.append(detection)

                if not valid_detections:
                    logger.error(f"No valid detections for frame {frame_id}.")
                    continue

                embs = extractor(cropped_imgs)
                if len(embs) != len(valid_detections):
                    logger.error(f"Mismatch in detections and embeddings count for frame {frame_id}.")
                    continue

                embs = embs.cpu().detach().numpy()
                online_targets = tracker.update(np.array(valid_detections), embs)

                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.last_tlwh
                    tid = t.track_id
                    if tlwh[2] * tlwh[3] > args.min_box_area:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        results.append(
                            f"{frame_id+1},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                
                timer.toc()
                online_im = plot_tracking(
                    frame, online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / max(1e-5, timer.average_time)
                )
            else:
                timer.toc()
                online_im = frame

        except Exception as e:
            logger.error(f"Error processing frame {frame_id} from {img_path}: {str(e)}")
            continue

        if args.save_result:
            vid_writer.write(online_im)

        frame_id += 1

    if args.save_result:
        res_file = osp.join(save_folder, video_id + "_results.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")

    vid_writer.release()
    gc.collect()

class MAML:
    def __init__(self, num_sports, max_classes, num_inner_steps, inner_lr, learn_inner_lrs, outer_lr, log_dir, device, model):
        self.model = model
        self.num_sports = num_sports
        self.max_classes = max_classes
        self.device = device
        self._num_inner_steps = num_inner_steps
        # self._inner_lrs = inner_lr  
        self._outer_lr = outer_lr
        self._optimizer = torch.optim.Adam(model.parameters(), lr=self._outer_lr)
        self._log_dir = log_dir
        os.makedirs(self._log_dir, exist_ok=True)
        self._start_train_step = 0

        # self.inner_lr_dict = {name: self._inner_lrs for name, param in self.model.named_parameters()}
        # self._inner_lrs = {k: torch.tensor(inner_lr, requires_grad=learn_inner_lrs) for k in self.model.named_parameters()}
        self._inner_lrs = {name: torch.tensor(inner_lr, requires_grad=learn_inner_lrs) for name, _ in self.model.named_parameters()}


    def _forward(self, images):
        return self.model(images)
    
    def get_player_id_logits(self, images, train, isQuery=False):
        """
        Accumulates logits from a model for a batch of images.

        Args:
            images (torch.Tensor): The input images.

        Returns:
            torch.Tensor: The accumulated logits. of shape: (num_support, num_sports, num_way)
        """
        player_id_logits = torch.Tensor([]).to(args.device)
        for sport_idx in range(images.shape[2]):
            # Iterate over each video in the sport
            for video_idx in range(images.shape[1]):
                # Initialize a list to hold all frames of the current video

                # Convert the list of frames to a tensor
                video_frames_tensor = images[:, video_idx, sport_idx]
                video_frames_tensor = video_frames_tensor.reshape(-1, 3, int(720 * 0.25), int(1280 * 0.25))
                video_frames_tensor.to(args.device)

                # Process the batch of frames through the model
                logits = self.model(video_frames_tensor)
                player_id_logits = torch.cat((player_id_logits, logits), dim=0) # output shape: (num frames*num_sports, num_way)

        if isQuery:
            player_id_logits = player_id_logits.reshape(args.num_query, args.num_sports, args.num_way)

        else:
            player_id_logits = player_id_logits.reshape(args.num_support, args.num_sports, args.num_way)


        return player_id_logits

    def _inner_loop(self, images, labels, train):

        accuracies = []
        accuracy_dict = {}
        accuracy_dict2 = {}
        # parameters = {name: param.clone() for name, param in self.model.named_parameters()}
        parameters = list(self.model.parameters())

        for step in range(self._num_inner_steps):
            self.model.train() if train else self.model.eval()


            player_id_logits = self.get_player_id_logits(images, train)
            player_id_logits.to(args.device)


            labels = labels.squeeze(1)
            player_id_labels = labels[..., 1]  # shape (num frames, num sports, num way)
            player_id_labels.to(args.device)
            player_id_labels = player_id_labels + 1

            if ONE_ZERO_LABELS:
                player_id_labels[player_id_labels[:,:] != 0] = 1
            player_id_loss = F.cross_entropy(player_id_logits, player_id_labels)



            # Calculate accuracy for each component
            accuracy_dict = {
            "player_id": util.calculate_accuracy(player_id_logits, player_id_labels, ONE_ZERO_LABELS)
            }
            accuracies.append(accuracy_dict)


            if train:
                gradients = torch.autograd.grad(player_id_loss, parameters, create_graph=True, allow_unused=True)
            else:
                gradients = torch.autograd.grad(player_id_loss, parameters, create_graph=False)

            with torch.no_grad(): 
                for param, grad, (name, _) in zip(parameters, gradients, self.model.named_parameters()):
                    if grad is not None:
                        # Retrieve the learning rate for the current parameter from the dictionary
                        lr = self._inner_lrs[name].item()  
                        # Update the parameter using its gradient and learning rate
                        param.data -= lr * grad

        
        player_id_logits2 = self.get_player_id_logits(images, train)

            # Calculate accuracy for each component
        accuracy_dict2  = {
            "player_id": util.calculate_accuracy(player_id_logits2, player_id_labels, ONE_ZERO_LABELS),
            }
        accuracies.append(accuracy_dict2)
        # print(accuracies)
        if train:
            return accuracies
        else:
            return accuracies, self.model.state_dict(), self._optimizer.state_dict()


    def _outer_step(self, images, labels, train, f1_score=None, acc_score=None):
        """Computes the MAML loss and metrics on a batch of tasks.

        Args:
            images (Tensor): Batch of images for all tasks, shape (K+num_query, num_videos, num_sports, rgb_image_size)
            labels (Tensor): Batch of labels for all tasks, shape (K+num_query, num_videos, num_sports, max_number_players_on_screen, 6)
            train (bool): Whether we are training or evaluating

        Returns:
            outer_loss (Tensor): Mean MAML loss over the batch, scalar
            accuracies_support (ndarray): Support set accuracy over the course of the inner loop, averaged over the task batch, shape (num_inner_steps + 1,)
            accuracy_query (float): Query set accuracy of the adapted parameters, averaged over the task batch
        """
        outer_loss_batch = []
        accuracies_support_batch = []
        accuracy_query_batch = []

        if args.test:
            batch_size = args.test_batch_size
        else:
            batch_size = args.meta_batch_size
 
        for task_idx in range(batch_size):  # Iterate over tasks in the batch
            images_task = images[task_idx]  # Images for the current task
            labels_task = labels[task_idx]  # Labels for the current task

            images_support = images_task[:args.num_support] 
            labels_support = labels_task[:args.num_support] 
            images_query = images_task[args.num_support:]  
            labels_query = labels_task[args.num_support:]   

            images_support = images_support.to(self.device)
            labels_support = labels_support.to(self.device)
            images_query = images_query.to(self.device)
            labels_query = labels_query.to(self.device)

            if train:
                accuracies_support = self._inner_loop(images_support, labels_support, train)
            else:
                accuracies_support, model_inner_state, optimizer_inner_state = self._inner_loop(images_support, labels_support, train)


            
            # Initialize lists to store losses and accuracies for each output component
            accuracy_query = []

            # print("labels_query:", labels_query.shape)
            labels_query = labels_query.squeeze(1)
            player_id_labels = labels_query[..., 1]
            player_id_labels = player_id_labels + 1
            player_id_labels.to(args.device)

            player_id_logits = self.get_player_id_logits(images_query, train, isQuery=True)

            if ONE_ZERO_LABELS:
                player_id_logits[player_id_labels[:,:] != 0] = 1
            player_id_loss = F.cross_entropy(player_id_logits, player_id_labels)
        
            # Calculate accuracy for each component
            accuracy_query = {
            "player_id": util.calculate_accuracy(player_id_logits, player_id_labels, ONE_ZERO_LABELS)
            }
            if not train:
                _, f1_score, acc_score = util.calculate_accuracy_and_f1(player_id_logits, player_id_labels, f1_score, acc_score, ONE_ZERO_LABELS)

            outer_loss_batch.append(player_id_loss)
            accuracies_support_batch.append(accuracies_support)
            accuracy_query_batch.append(accuracy_query)

        outer_loss = torch.mean(torch.stack(outer_loss_batch))

        # Initialize dictionaries to accumulate accuracy values for each subtask
        accumulated_pre_adapt_accuracies = {
            "player_id": []
        }

        accumulated_post_adapt_accuracies = {
            "player_id": []
        }

        # Iterate through the list of dictionaries and accumulate accuracy values
        for key in accumulated_pre_adapt_accuracies:
            # Calculate the mean accuracy for the current subtask key
            mean_pre_adapt_accuracy = np.mean([item[key] for item in accuracies_support_batch[0]])  # Index 0 for pre-adaptation
            mean_post_adapt_accuracy = np.mean([item[key] for item in accuracies_support_batch[-1]])  # Index -1 for post-adaptation
            accumulated_pre_adapt_accuracies[key].append(mean_pre_adapt_accuracy)
            accumulated_post_adapt_accuracies[key].append(mean_post_adapt_accuracy)

        # Convert the accumulated accuracy values to NumPy arrays and calculate the mean accuracy for each subtask
        pre_adapt_accuracy_mean = {
            key: np.mean(np.array(accumulated_pre_adapt_accuracies[key]), axis=0) for key in accumulated_pre_adapt_accuracies
        }

        post_adapt_accuracy_mean = {
            key: np.mean(np.array(accumulated_post_adapt_accuracies[key]), axis=0) for key in accumulated_post_adapt_accuracies
        }


        accumulated_accuracies = {
            "player_id": []
        }

        # Iterate through the list of dictionaries and accumulate accuracy values
        for acc in accuracy_query_batch:
            for key in accumulated_accuracies:
                # Calculate the mean accuracy for the current subtask key
                mean_accuracy = acc[key]  
                accumulated_accuracies[key].append(mean_accuracy)

        # Convert the accumulated accuracy values to NumPy arrays and calculate the mean accuracy for each subtask
        accuracy_query_mean = {
            key: np.mean(np.array(accumulated_accuracies[key]), axis=0) for key in accumulated_accuracies
        }

        # print(f" query accuracies: {accuracy_query_mean}")

        # print("\nOuter loss: ", outer_loss)
        if train:
            return outer_loss, pre_adapt_accuracy_mean, post_adapt_accuracy_mean, accuracy_query_mean, 
        else:
            return outer_loss, pre_adapt_accuracy_mean, post_adapt_accuracy_mean, accuracy_query_mean, model_inner_state, optimizer_inner_state, f1_score, acc_score


    def train(self, dataloader_meta_train, dataloader_meta_val, writer):
        """Train the MAML.

        Consumes dataloader_meta_train to optimize MAML meta-parameters
        while periodically validating on dataloader_meta_val, logging metrics, and
        saving checkpoints.

        Args:
            dataloader_meta_train (DataLoader): loader for train tasks
            dataloader_meta_val (DataLoader): loader for validation tasks
            writer (SummaryWriter): TensorBoard logger
        """
        torch.autograd.set_detect_anomaly(True)
        best_val_accuracy = float('-inf')
        best_f1 = float('-inf')

        last_best_step = 0 
        i = 0
        divisor = args.meta_batch_size_miguel // args.meta_batch_size
        f1_score = MulticlassF1Score(num_classes=NUM_CLASSES_FRAME_ID)
        acc_score = MulticlassAccuracy(num_classes=NUM_CLASSES_FRAME_ID)
        if os.getlogin() == "darke" and args.meta_batch_size_miguel % args.meta_batch_size != 0 :
            raise ValueError(f"batch sizes are not divisable.  {args.meta_batch_size_miguel}, {args.meta_batch_size}")
        print(f'Starting training at iteration {self._start_train_step}.')
        for i_step, (images, labels, sports_order) in tqdm(enumerate(dataloader_meta_train, start=self._start_train_step), total=args.meta_train_iterations):
            i += 1
            if i_step >= args.meta_train_iterations:
                break

            self.model.train()
            outer_loss, pre_adapt_accuracy_mean, post_adapt_accuracy_mean, accuracy_query = (
                self._outer_step(images, labels, train=True)
            )
            print(f"Outer Loss: {outer_loss}")

            if math.isnan(outer_loss.item()):
                raise ValueError
            
            if os.getlogin() == "darke":
                outer_loss /= divisor
                outer_loss.backward()  
                if i % divisor == 0:
                    self._optimizer.step()
                    self._optimizer.zero_grad()
                    i = 0
            else:
                outer_loss.backward()  
                self._optimizer.step()
                self._optimizer.zero_grad()


            # Write loss value to TensorBoard
            writer.add_scalar('loss/train', outer_loss.item(), i_step)

            # Write pre-adaptation support accuracies to TensorBoard
            for key, value in pre_adapt_accuracy_mean.items():
                writer.add_scalar(f'train_accuracy/pre_adapt_support/{key}', value, i_step)

            # Write post-adaptation support accuracies to TensorBoard
            for key, value in post_adapt_accuracy_mean.items():
                writer.add_scalar(f'train_accuracy/post_adapt_support/{key}', value, i_step)

            # Write post-adaptation query accuracies to TensorBoard
            for key, value in accuracy_query.items():
                writer.add_scalar(f'train_accuracy/post_adapt_query/{key}', value, i_step)


            # print("VALIDATION")
            if i_step != 0 and i_step % VAL_INTERVAL == 0:
                self.model.eval()
                losses = []
                val_accuracies_pre_adapt_support = {}
                val_accuracies_post_adapt_support = {}
                val_accuracies_post_adapt_query = {}

                # Iterate through validation batches
                for batch_idx, val_batch in enumerate(dataloader_meta_val):
                    # torch.cuda.empty_cache()
                    if batch_idx >= args.meta_val_iterations:
                        break
                    # print(f"Batch {batch_idx + 1}/{args.meta_val_iterations}")
                    val_images, val_labels, _ = val_batch  # Unpack the batch

                    val_outer_loss, val_pre_adapt_accuracy_mean, val_post_adapt_accuracy_mean, val_accuracy_query, model_inner_state, optimizer_inner_state,  f1_score, acc_score = (
                        self._outer_step(val_images, val_labels, train=False, f1_score=f1_score, acc_score=acc_score)
                    )
                    losses.append(val_outer_loss.item())

                    # Collect pre-adaptation support accuracies
                    for key, value in val_pre_adapt_accuracy_mean.items():
                        if key not in val_accuracies_pre_adapt_support:
                            val_accuracies_pre_adapt_support[key] = []
                        val_accuracies_pre_adapt_support[key].append(value)

                    # Collect post-adaptation support accuracies
                    for key, value in val_post_adapt_accuracy_mean.items():
                        if key not in val_accuracies_post_adapt_support:
                            val_accuracies_post_adapt_support[key] = []
                        val_accuracies_post_adapt_support[key].append(value)

                    # Collect post-adaptation query accuracies
                    for key, value in val_accuracy_query.items():
                        if key not in val_accuracies_post_adapt_query:
                            val_accuracies_post_adapt_query[key] = []
                        val_accuracies_post_adapt_query[key].append(value)

                # Calculate mean loss and accuracies over validation batches
                val_mean_loss = np.mean(losses)
                
                val_mean_accuracies_pre_adapt_support = {}
                for key, value_list in val_accuracies_pre_adapt_support.items():
                    val_mean_accuracies_pre_adapt_support[key] = np.mean(value_list)

                val_mean_accuracies_post_adapt_support = {}
                for key, value_list in val_accuracies_post_adapt_support.items():
                    val_mean_accuracies_post_adapt_support[key] = np.mean(value_list)

                val_mean_accuracies_post_adapt_query = {}
                for key, value_list in val_accuracies_post_adapt_query.items():
                    val_mean_accuracies_post_adapt_query[key] = np.mean(value_list)

                # Write validation results to TensorBoard
                writer.add_scalar('loss/val', val_mean_loss, i_step)

                for key, value in val_mean_accuracies_pre_adapt_support.items():
                    writer.add_scalar(f'val_accuracy/val_pre_adapt_support/{key}', value, i_step)

                for key, value in val_mean_accuracies_post_adapt_support.items():
                    writer.add_scalar(f'val_accuracy/val_post_adapt_support/{key}', value, i_step)

                for key, value in val_mean_accuracies_post_adapt_query.items():
                    writer.add_scalar(f'val_accuracy/val_post_adapt_query/{key}', value, i_step)

                current_val_accuracy = np.mean([val for val in val_mean_accuracies_post_adapt_query.values()])
                
                f1 = f1_score.compute().item()
                ac = acc_score.compute().item()

                print("Best Accuracy: ", best_val_accuracy)
                print("current accuracy: ", current_val_accuracy)
                print("acc from torch eval: ", ac)
                print("Best f1 score: ", best_f1)
                print("current f1 score: ", f1)


                # Check if the current validation accuracy is better than the best one seen so far
                if current_val_accuracy > best_val_accuracy or f1 > best_f1:
                    print("new best! at i_step: ", i_step, "trigger by f1" if f1 > best_f1 else "triggered by acc")
                    print("last best accuracy: ", best_val_accuracy)
                    #print("last best f1: ", best_f1)
                    if current_val_accuracy > best_val_accuracy:
                        best_val_accuracy = current_val_accuracy
                        best_acc_step = i_step
                        print("NEW best accuracy: ", best_val_accuracy)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_f1_step = i_step
                        print("NEW best f1: ", best_f1)
                    last_best_step = i_step
                    self._save(i_step, model_inner_state, optimizer_inner_state)  # Save the model checkpoint
                    with open(self._log_dir + "/final_accuracies.txt", "w") as F:
                        F.write(f"last best i_step: {last_best_step}\nlast best accuracy: {best_val_accuracy} at step: {best_acc_step}\nlast best f1: {best_f1} at step: {best_f1_step}")
                    print(f'Saved new best checkpoint with validation accuracy: {best_val_accuracy:.4f}')
            
                
            if i_step - last_best_step >= 100:
                print("Stopping training - 100 steps have passed since the last best accuracy")
                print("last best i_step: ", last_best_step)
                print("last best accuracy: ", best_val_accuracy)
                print("last best f1: ", best_f1)
                break



    def _load(self, checkpoint_step):
        """Loads parameters and optimizer state_dict from a checkpoint.

        Args:
            checkpoint_step (int): iteration of the checkpoint to load

        Raises:
            ValueError: if checkpoint for checkpoint_step is not found
        """
        checkpoint_path = f'{os.path.join(self._log_dir, "state")}{checkpoint_step}.pt'
        
        if not os.path.isfile(checkpoint_path):
            raise ValueError(f"No checkpoint found at '{checkpoint_path}'")

        state = torch.load(checkpoint_path)

        if 'model_state_dict' in state:
            self.model.load_state_dict(state['model_state_dict'])
            print("Model state loaded successfully.")

        if 'optimizer_state_dict' in state:
            self._optimizer.load_state_dict(state['optimizer_state_dict'])
            print("Optimizer state loaded successfully.")

        self._start_train_step = checkpoint_step + 1
        print(f'Loaded checkpoint iteration {checkpoint_step}.')


    def _save(self, checkpoint_step, model_inner_state, optimizer_inner_state):
        """Saves parameters and optimizer state_dict as a checkpoint.

        Args:
            checkpoint_step (int): iteration to label checkpoint with
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),  
            'optimizer_state_dict': self._optimizer.state_dict(),  
            'checkpoint_step': checkpoint_step
        }, f'{os.path.join(self._log_dir, "state")}{checkpoint_step}.pt')
        torch.save({
            'model_state_dict': model_inner_state,  
            'optimizer_state_dict': optimizer_inner_state,  
            'checkpoint_step': checkpoint_step
        }, f'{os.path.join(self._log_dir, "inner_state")}{checkpoint_step}.pt')
        
        print('Saved checkpoint.')
        torch.cuda.empty_cache() 
        print('Cleared cuda cache.')   


def get_support_query(images, labels):
    support_frames = images[:,:-args.num_query]
    support_labels = labels[:,:-args.num_query]
    query_frames = images[:,-args.num_query:]
    query_labels = labels[:,-args.num_query:]
    support_frames = torch.squeeze(support_frames, dim=2)
    support_labels = torch.squeeze(support_labels, dim=2)
    query_frames = torch.squeeze(query_frames, dim=2)
    query_labels = torch.squeeze(query_labels, dim=2)
    return support_frames, support_labels, query_frames, query_labels

def print_support_query_shapes(support_frames, support_labels, query_frames, query_labels):
    print("\nSupport Set Shape:", support_frames.shape)
    print("Support Label Shape:", support_labels.shape)
    print("Query Set Shape:", query_frames.shape)
    print("Query Label Shape:", query_labels.shape)

def find_latest_checkpoint(log_dir):
    # Create a regex pattern to match the state files with numbers
    pattern = re.compile(r"state(\d+)\.pt")

    # List all files in the log directory and filter out non-checkpoint files
    checkpoints = [f for f in os.listdir(log_dir) if pattern.match(f)]

    if not checkpoints:
        return None

    # Find the checkpoint with the largest number in its filename
    latest_checkpoint = max(checkpoints, key=lambda x: int(pattern.match(x).group(1)))

    # Extract the largest number
    largest_number = int(pattern.match(latest_checkpoint).group(1))
    print(f"Largest checkpoint number found: {largest_number}")

    return os.path.join(log_dir, latest_checkpoint)

def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    vis_folder = output_dir
    os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")
    
    logger.info("Using device: {}".format(args.device))
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    log_dir = args.log_dir
    if log_dir is None:
        if os.getlogin() == "darke":
            log_dir = f"D:/logs/{args.num_way}.support_{args.num_support}.query_{args.num_query}.inner_steps_{args.num_inner_steps}.inner_lr_{args.inner_lr}.learn_inner_lrs_{args.learn_inner_lrs}.outer_lr_{args.outer_lr}.batch_size_{args.meta_batch_size}.train_iter_{args.meta_train_iterations}..val_iter_{args.meta_val_iterations}"  
        else:
            log_dir = f"logs/{args.model}/{args.experiment_name}/.Test_{args.test}.No_Pretrained_weights.way_{args.num_way}.support_{args.num_support}.query_{args.num_query}.inner_steps_{args.num_inner_steps}.inner_lr_{args.inner_lr}.learn_inner_lrs_{args.learn_inner_lrs}.outer_lr_{args.outer_lr}.batch_size_{args.meta_batch_size}.train_iter_{args.meta_train_iterations}..val_iter_{args.meta_val_iterations}"  
    logger.info(f"Run parameters {log_dir}")
    writer = tensorboard.SummaryWriter(log_dir)

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    model.eval()
    if not args.trt:
        if args.ckpt is None:
            ckpt_file = "checkpoints/best_ckpt.pth.tar"
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None


    extractor_model = build_model(
    name='osnet_x1_0', 
    num_classes= 23, 
    loss='softmax',  
    pretrained=True, 
    use_gpu=torch.cuda.is_available()
    )

    model_path = 'checkpoints/sports_model.pth.tar-60'


    # Load pretrained weights
    if model_path and check_isfile(model_path):
        load_pretrained_weights(extractor_model, model_path)


    extractor_model.to(args.device)

    extractor_model.train()
    # Fine Tune
    if args.fine_tune and args.model == 'maml':
        maml = MAML(args.num_sports, args.num_way, args.num_inner_steps, args.inner_lr,
                    args.learn_inner_lrs, args.outer_lr, log_dir, args.device, extractor_model)

        if args.checkpoint_step > -1:
            maml.load(args.checkpoint_step)
        else:
            logger.info('Checkpoint loading skipped.')

        if not args.test:
            meta_train_iterable = DataGenerator(
            args.num_videos,
            args.num_support + args.num_query,
            batch_type="train",
            cache=False,
            generate_new_tasks=True,
            config={"data_folder":base_path + "/data_cs"}
            )
            meta_train_loader = iter(
                torch.utils.data.DataLoader(
                    meta_train_iterable,
                    batch_size=args.meta_batch_size,
                    num_workers=args.num_workers,
                    # pin_memory=True,
                )
            )

            meta_val_iterable = DataGenerator(
            args.num_videos,
            args.num_support + args.num_query,
            batch_type="val",
            cache=False,
            generate_new_tasks=True,
            config={"data_folder":base_path + "/data_cs"}
            )
            meta_val_loader = iter(
                torch.utils.data.DataLoader(
                    meta_val_iterable,
                    batch_size=args.meta_batch_size,
                    num_workers=args.num_workers,
                    # pin_memory=True,
                )
            )

            print("Loaded data")
            maml.train(meta_train_loader, meta_val_loader, writer)

    elif args.fine_tune:
        logger.error(f"ERROR: Model '{args.model}' is not implemented for fine-tuning")
        return
    


    latest_checkpoint_path = find_latest_checkpoint(log_dir)
    if latest_checkpoint_path:
        logger.info(f"Loading fine tuned checkpoint: {latest_checkpoint_path}")
        checkpoint = torch.load(latest_checkpoint_path, map_location=args.device)
        extractor_model.load_state_dict(checkpoint['model_state_dict'])
        print("Fine tuned checkpoint loaded successfully.")
    else:
        logger.error("No fine tuned checkpoints found.")

    extractor_model.eval()
    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    extractor = FeatureExtractor(model=extractor_model, device='cuda') 
    # extractor = FeatureExtractor(model=extractor_model, model_path = 'checkpoints/sports_model.pth.tar-60', device='cuda')   


    dir = args.path  
    print(args.path)
    if os.path.exists(os.path.join(dir, 'img1')):
        process_sequence(predictor, extractor, os.path.join(dir, 'img1'), vis_folder, args)
    else:
        # Otherwise, assume it contains subdirectories, each representing a video sequence
        sequence_dirs = [os.path.join(dir, f) for f in os.listdir(dir) if os.path.isdir(os.path.join(dir, f))]

        for seq_dir in sequence_dirs:
            img_dir = os.path.join(seq_dir, 'img1')
            if os.path.exists(img_dir):
                process_sequence(predictor, extractor, img_dir, vis_folder, args)
            else:
                logger.error(f"'img1' subfolder not found in {seq_dir}")
    process_results(args.experiment_name, base_path)

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    args.experiment_name = f"{args.num_way}.support_{args.num_support}.query_{args.num_query}.inner_steps_{args.num_inner_steps}.inner_lr_{args.inner_lr}.learn_inner_lrs_{args.learn_inner_lrs}.outer_lr_{args.outer_lr}.batch_size_{args.meta_batch_size}.train_iter_{args.meta_train_iterations}..val_iter_{args.meta_val_iterations}"
    main(exp, args)
