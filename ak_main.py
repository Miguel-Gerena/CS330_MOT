from data import DataGenerator  
from ak_util import load_by_sport
import util
import sys
import argparse
import os
import random
import json
import numpy as np
import torch
import torch.multiprocessing
import multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader, BatchSampler, RandomSampler
from torch import nn
import torch.nn.functional as F
from torch import autograd
from torch.utils import tensorboard
from motmetrics.metrics import motp, motp
from collections import OrderedDict

NUM_INPUT_CHANNELS = 1
NUM_HIDDEN_CHANNELS = 10 # usually 32
KERNEL_SIZE = 3
NUM_CONV_LAYERS = 2     #usually 4
SUMMARY_INTERVAL = 10
SAVE_INTERVAL = 100
LOG_INTERVAL = 10
VAL_INTERVAL = LOG_INTERVAL * 5
NUM_TEST_TASKS = 600
NUM_CLASSES_FRAME_ID = 23  # Number of classes for frame ID
NUM_CLASSES_PLAYER_ID = 23  # Number of classes for player ID

class MAML:
    def __init__(self, num_outputs, num_inner_steps, inner_lr, learn_inner_lrs, outer_lr, log_dir, device):
        meta_parameters = {}

        self.device = device

        # Construct feature extractor
        in_channels = NUM_INPUT_CHANNELS
        for i in range(NUM_CONV_LAYERS):
            meta_parameters[f'conv{i}'] = nn.init.xavier_uniform_(
                torch.empty(NUM_HIDDEN_CHANNELS, in_channels, KERNEL_SIZE, KERNEL_SIZE, requires_grad=True, device=self.device)
            )
            meta_parameters[f'b{i}'] = nn.init.zeros_(
                torch.empty(NUM_HIDDEN_CHANNELS, requires_grad=True, device=self.device)
            )
            in_channels = NUM_HIDDEN_CHANNELS

        # Construct linear head layer for frame ID and player ID
        # meta_parameters['w_frame_id'] = nn.init.xavier_uniform_(
        #     torch.empty(NUM_CLASSES_FRAME_ID, NUM_HIDDEN_CHANNELS, requires_grad=True, device=self.device)
        # )
        # meta_parameters['b_frame_id'] = nn.init.zeros_(
        #     torch.empty(NUM_CLASSES_FRAME_ID, requires_grad=True, device=self.device)
        # )
        meta_parameters['w_player_id'] = nn.init.xavier_uniform_(
            torch.empty(NUM_CLASSES_PLAYER_ID, NUM_HIDDEN_CHANNELS, requires_grad=True, device=self.device)
        )
        meta_parameters['b_player_id'] = nn.init.zeros_(
            torch.empty(NUM_CLASSES_PLAYER_ID, requires_grad=True, device=self.device)
        )

        # Construct linear layers for bbox components
        for bbox_component in ['left', 'top', 'width', 'height']:
            meta_parameters[f'w_bbox_{bbox_component}'] = nn.init.xavier_uniform_(
                torch.empty(NUM_CLASSES_PLAYER_ID, NUM_HIDDEN_CHANNELS, requires_grad=True, device=self.device)
            )
            meta_parameters[f'b_bbox_{bbox_component}'] = nn.init.zeros_(
                torch.empty(NUM_CLASSES_PLAYER_ID, requires_grad=True, device=self.device)
            )

        self._meta_parameters = meta_parameters
        self._num_inner_steps = num_inner_steps
        self._inner_lrs = {k: torch.tensor(inner_lr, requires_grad=learn_inner_lrs) for k in self._meta_parameters.keys()}
        self._outer_lr = outer_lr

        self._optimizer = torch.optim.Adam(
            list(self._meta_parameters.values()) + list(self._inner_lrs.values()),
            lr=self._outer_lr
        )
        self._log_dir = log_dir
        os.makedirs(self._log_dir, exist_ok=True)

        self._start_train_step = 0

    def _forward(self, images, parameters):
        x = images
        for i in range(NUM_CONV_LAYERS):
            x = F.conv2d(input=x, weight=parameters[f'conv{i}'], bias=parameters[f'b{i}'], stride=1, padding="same")
            x = F.batch_norm(x, None, None, training=True)
            x = F.relu(x)
        x = torch.mean(x, dim=[-1, -2])  # Global average pooling

        # Frame ID logits (assuming a classification task for frame ID)
        # frame_id_logits = F.linear(input=x, weight=parameters['w_frame_id'], bias=parameters['b_frame_id'])

        # Player ID logits - now just indexes within a frame, not unique IDs
        player_id_logits = F.linear(input=x, weight=parameters['w_player_id'], bias=parameters['b_player_id'])
        player_id_logits = player_id_logits.view(-1, NUM_CLASSES_PLAYER_ID)  # Shape: [batch_size, max_players]

        # Bbox components for each potential player
        bb_left = F.linear(input=x, weight=parameters['w_bbox_left'], bias=parameters['b_bbox_left'])
        bb_top = F.linear(input=x, weight=parameters['w_bbox_top'], bias=parameters['b_bbox_top'])
        bb_width = F.linear(input=x, weight=parameters['w_bbox_width'], bias=parameters['b_bbox_width'])
        bb_height = F.linear(input=x, weight=parameters['w_bbox_height'], bias=parameters['b_bbox_height'])


        return player_id_logits, bb_left, bb_top, bb_width, bb_height

    def _inner_loop(self, images, labels, train):

        accuracies = []
        accuracy_dict = {}
        parameters = {
            k: torch.clone(v)
            for k, v in self._meta_parameters.items()
        }
        gradients = None
        num_inner_steps = self._num_inner_steps
        
        for step in range(num_inner_steps):
            outputs = self._forward(images, parameters)
            player_id_logits, bb_left_logits, bb_top_logits, bb_width_logits, bb_height_logits = outputs

            # print("Logits:")
            # print(f"player id {player_id_logits.shape}")
            # print(f"bb left {bb_left_logits.shape}")
            # print(f"bb top {bb_top_logits.shape}")
            # print(f"bb width {bb_width_logits.shape}")
            # print(f"bb height {bb_height_logits.shape}")
        
            # print("labels:", labels.shape)
            labels = labels[:, 0, 0, :]


            frame_id_labels = labels[..., 0]
            player_id_labels = labels[..., 1]
            bb_left_labels = labels[..., 2]
            bb_top_labels = labels[..., 3]
            bb_width_labels = labels[..., 4]
            bb_height_labels = labels[..., 5]



            # print("labels:")
            # print(f"player id {player_id_labels.shape}")
            # print(f"bb left {bb_left_labels.shape}")
            # print(f"bb top {bb_top_labels.shape}")
            # print(f"bb width {bb_width_labels.shape}")
            # print(f"bb height {bb_height_labels.shape}")


            # Calculate losses for each component
            # frame_id_loss = F.cross_entropy(frame_id_logits, frame_id_labels)
            player_id_loss = F.cross_entropy(player_id_logits, player_id_labels, ignore_index=-1)
            bb_left_loss = F.mse_loss(bb_left_logits, bb_left_labels)
            bb_top_loss = F.mse_loss(bb_top_logits, bb_top_labels)
            bb_width_loss = F.mse_loss(bb_width_logits, bb_width_labels)
            bb_height_loss = F.mse_loss(bb_height_logits, bb_height_labels)

            loss = (player_id_loss + bb_left_loss + bb_top_loss + bb_width_loss + bb_height_loss)/5

            # weight_bbox = 2.0  # Higher weight for bbox losses
            # weight_classification = 1.0             
            # loss = (weight_classification * player_id_loss + weight_bbox * (bb_left_loss + bb_top_loss + bb_width_loss + bb_height_loss)) / (weight_classification + 4 * weight_bbox)
            print("loss:", loss)

            predicted_bboxes = torch.stack([bb_left_logits, bb_top_logits, bb_width_logits, bb_height_logits], dim=-1)
            true_bboxes = torch.stack([bb_left_labels, bb_top_labels, bb_width_labels, bb_height_labels], dim=-1)

            # Calculate accuracy for each component
            accuracy_dict = {
            "player_id": util.calculate_accuracy(player_id_logits, player_id_labels),
            "bbox": util.calculate_iou(predicted_bboxes, true_bboxes)
            }
            accuracies.append(accuracy_dict)
            # print(f"Accuracies : {accuracy_dict}")

            # Calculate gradients using autograd.grad
            if train: 
                gradients = torch.autograd.grad(loss, parameters.values(), create_graph=True)
            else:
                gradients = torch.autograd.grad(loss, parameters.values(), create_graph=False)

            # Update parameters using gradient descent with individual inner learning rates
            for idx, key in enumerate(parameters.keys()):
                parameters[key] = parameters[key] - self._inner_lrs[key] * gradients[idx] 

        updated_outputs = self._forward(images, parameters)
        player_id_logits2, bb_left_logits2, bb_top_logits2, bb_width_logits2, bb_height_logits2 = updated_outputs

        # print("Logits2:")
        # print(f"player id {player_id_logits2.shape}")
        # print(f"bb left {bb_left_logits2.shape}")
        # print(f"bb top {bb_top_logits2.shape}")
        # print(f"bb width {bb_width_logits2.shape}")
        # print(f"bb height {bb_height_logits2.shape}")

        predicted_bboxes = torch.stack([bb_left_logits2, bb_top_logits2, bb_width_logits2, bb_height_logits2], dim=-1)

            # Calculate accuracy for each component
        accuracy_dict2 = {
            "player_id": util.calculate_accuracy(player_id_logits, player_id_labels),
            "bbox": util.calculate_iou(predicted_bboxes, true_bboxes)
            }
        accuracies.append(accuracy_dict2)
        # print(f"Accuracies 2: {accuracy_dict2}")

        return parameters, accuracies, gradients

    def _outer_step(self, images, labels, train):
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

        for task_idx in range(args.meta_batch_size):  # Iterate over tasks in the batch
            print("task num:", task_idx+1)
            images_task = images[task_idx]  # Images for the current task
            labels_task = labels[task_idx]  # Labels for the current task

            images_support = images_task[:args.num_support] 
            labels_support = labels_task[:args.num_support] 
            images_query = images_task[args.num_support:]  
            labels_query = labels_task[args.num_support:]   
            
            # print("\nSupport Set Shape:", images_support.shape)
            # print("Support Label Shape:", labels_support.shape)
            # print("Query Set Shape:", images_query.shape)
            # print("Query Label Shape:", labels_query.shape)


            images_support = images_support.to(self.device)
            labels_support = labels_support.to(self.device)
            images_query = images_query.to(self.device)
            labels_query = labels_query.to(self.device)

            adapted_params, accuracies_support, _ = self._inner_loop(images_support, labels_support, train)
            logits_query = self._forward(images_query, adapted_params)
            
            # Initialize lists to store losses and accuracies for each output component
            accuracy_query = []
            player_id_logits, bb_left_logits, bb_top_logits, bb_width_logits, bb_height_logits = logits_query

            # print("Logits:")
            # print(f"player id {player_id_logits.shape}")
            # print(f"bb left {bb_left_logits.shape}")
            # print(f"bb top {bb_top_logits.shape}")
            # print(f"bb width {bb_width_logits.shape}")
            # print(f"bb height {bb_height_logits.shape}")

            # print("labels_query:", labels_query.shape)
            labels_query = labels_query[:, 0, 0, :]

            frame_id_labels = labels_query[..., 0]
            player_id_labels = labels_query[..., 1]
            bb_left_labels = labels_query[..., 2]
            bb_top_labels = labels_query[..., 3]
            bb_width_labels = labels_query[..., 4]
            bb_height_labels = labels_query[..., 5]


            # print("labels:")
            # print(f"player id {player_id_labels.shape}")
            # print(f"bb left {bb_left_labels.shape}")
            # print(f"bb top {bb_top_labels.shape}")
            # print(f"bb width {bb_width_labels.shape}")
            # print(f"bb height {bb_height_labels.shape}")


            player_id_loss = F.cross_entropy(player_id_logits, player_id_labels)
            bb_left_loss = F.mse_loss(bb_left_logits, bb_left_labels)
            bb_top_loss = F.mse_loss(bb_top_logits, bb_top_labels)
            bb_width_loss = F.mse_loss(bb_width_logits, bb_width_labels)
            bb_height_loss = F.mse_loss(bb_height_logits, bb_height_labels)

            loss = (player_id_loss + bb_left_loss + bb_top_loss + bb_width_loss + bb_height_loss)/5


            predicted_bboxes = torch.stack([bb_left_logits, bb_top_logits, bb_width_logits, bb_height_logits], dim=-1)
            true_bboxes = torch.stack([bb_left_labels, bb_top_labels, bb_width_labels, bb_height_labels], dim=-1)

            # Calculate accuracy for each component
            accuracy_query = {
            "player_id": util.calculate_accuracy(player_id_logits, player_id_labels),
            "bbox": util.calculate_iou(predicted_bboxes, true_bboxes)
            }
            # print(f"outer Accuracies : {accuracy_query}")

            outer_loss_batch.append(loss)
            accuracies_support_batch.append(accuracies_support)
            accuracy_query_batch.append(accuracy_query)


        outer_loss = torch.mean(torch.stack(outer_loss_batch))


        # Initialize dictionaries to accumulate accuracy values for each subtask
        accumulated_pre_adapt_accuracies = {
            "player_id": [],
             "bbox": []
        }

        accumulated_post_adapt_accuracies = {
            "player_id": [],
             "bbox": []
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

        print(f"Pre-adaptation accuracies: {pre_adapt_accuracy_mean}")
        print(f"Post-adaptation accuracies: {post_adapt_accuracy_mean}")



        accumulated_accuracies = {
            "player_id": [],
             "bbox": []
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

        print(f" query accuracies: {accuracy_query_mean}")

        return outer_loss, pre_adapt_accuracy_mean, post_adapt_accuracy_mean, accuracy_query_mean

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
        print(f'Starting training at iteration {self._start_train_step}.')
        for i_step, (images, labels, sports_order) in enumerate(dataloader_meta_train, start=self._start_train_step):
            if i_step >= args.meta_train_iterations:
                break
            print(f"Train Iteration: {i_step + 1}/{args.meta_train_iterations}")

            self._optimizer.zero_grad()
            outer_loss, pre_adapt_accuracy_mean, post_adapt_accuracy_mean, accuracy_query = (
                self._outer_step(images, labels, train=True)
            )
            outer_loss.backward()
            self._optimizer.step()

            print(f' Outer Loss: {outer_loss.item():.3f}')

            # Print pre-adaptation accuracies
            # print('Pre-adaptation support accuracies:')
            # for key, value in pre_adapt_accuracy_mean.items():
            #     print(f'{key}: {value:.3f}')

            # # Print post-adaptation accuracies
            # print('Post-adaptation support accuracies:')
            # for key, value in post_adapt_accuracy_mean.items():
            #     print(f'{key}: {value:.3f}')

            # # Print query accuracies
            # print('Query accuracies:')
            # for key, value in accuracy_query.items():
            #     print(f'{key}: {value:.3f}')

            # Write loss value to TensorBoard
            writer.add_scalar('loss/val', outer_loss.item(), i_step)

            # Write pre-adaptation support accuracies to TensorBoard
            for key, value in pre_adapt_accuracy_mean.items():
                writer.add_scalar(f'train_accuracy/pre_adapt_support/{key}', value, i_step)

            # Write post-adaptation support accuracies to TensorBoard
            for key, value in post_adapt_accuracy_mean.items():
                writer.add_scalar(f'train_accuracy/post_adapt_support/{key}', value, i_step)

            # Write post-adaptation query accuracies to TensorBoard
            for key, value in accuracy_query.items():
                writer.add_scalar(f'train_accuracy/post_adapt_query/{key}', value, i_step)


            
            if i_step % VAL_INTERVAL == 0:
                print("\nIn val interval check")
                losses = []
                val_accuracies_pre_adapt_support = {}
                val_accuracies_post_adapt_support = {}
                val_accuracies_post_adapt_query = {}

                # Iterate through validation batches
                for batch_idx, val_batch in enumerate(dataloader_meta_val):
                    if batch_idx >= args.meta_val_iterations:
                        break
                    print(f"Batch {batch_idx + 1}/{args.meta_val_iterations}")
                    val_images, val_labels, _ = val_batch  # Unpack the batch
                    # val_labels = val_labels[:, 0,0, :]

                    val_outer_loss, val_pre_adapt_accuracy_mean, val_post_adapt_accuracy_mean, val_accuracy_query = (
                        self._outer_step(val_images, val_labels, train=False)
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

                print("\n\nDONE WITH CHECK")
                

            if i_step % SAVE_INTERVAL == 0:
                self._save(i_step)


    # def test(self, dataloader_test):
    #     """Evaluate the MAML on test tasks.

    #     Args:
    #         dataloader_test (DataLoader): loader for test tasks
    #     """
    #     accuracies = []
    #     for task_batch in dataloader_test:
    #         _, _, accuracy_query = self._outer_step(task_batch, train=False)
    #         accuracies.append(accuracy_query)
    #     mean = np.mean(accuracies)
    #     std = np.std(accuracies)
    #     mean_95_confidence_interval = 1.96 * std / np.sqrt(NUM_TEST_TASKS)
    #     print(
    #         f'Accuracy over {NUM_TEST_TASKS} test tasks: '
    #         f'mean {mean:.3f}, '
    #         f'95% confidence interval {mean_95_confidence_interval:.3f}'
    #     )


    # def test(self, dataloader_test):
    #     """Evaluate the MAML on test tasks using MOT metrics.

    #     Args:
    #         dataloader_test (DataLoader): Loader for test tasks.
    #     """
    #     mota_scores = []
    #     motp_scores = []

    #     for task_batch in dataloader_test:
    #         mota, motp = compute_mot_metrics(task_batch)
            
    #         mota_scores.append(mota)
    #         motp_scores.append(motp)

    #     mean_mota = np.mean(mota_scores)
    #     std_mota = np.std(mota_scores)
    #     mean_motp = np.mean(motp_scores)
    #     std_motp = np.std(motp_scores)

    #     mota_confidence_interval = 1.96 * std_mota / np.sqrt(len(mota_scores))
    #     motp_confidence_interval = 1.96 * std_motp / np.sqrt(len(motp_scores))

    #     print(
    #         f'MOTA over {len(mota_scores)} test tasks: '
    #         f'mean {mean_mota:.3f}, '
    #         f'95% confidence interval {mota_confidence_interval:.3f}'
    #     )
        
    #     print(
    #         f'MOTP over {len(motp_scores)} test tasks: '
    #         f'mean {mean_motp:.3f}, '
    #         f'95% confidence interval {motp_confidence_interval:.3f}'
    #     )



    def load(self, checkpoint_step):
        """Loads a checkpoint.

        Args:
            checkpoint_step (int): iteration of checkpoint to load

        Raises:
            ValueError: if checkpoint for checkpoint_step is not found
        """
        target_path = (
            f'{os.path.join(self._log_dir, "state")}'
            f'{checkpoint_step}.pt'
        )
        if os.path.isfile(target_path):
            state = torch.load(target_path)
            self._meta_parameters = state['meta_parameters']
            self._inner_lrs = state['inner_lrs']
            self._optimizer.load_state_dict(state['optimizer_state_dict'])
            self._start_train_step = checkpoint_step + 1
            print(f'Loaded checkpoint iteration {checkpoint_step}.')
        else:
            raise ValueError(
                f'No checkpoint for iteration {checkpoint_step} found.'
            )

    def _save(self, checkpoint_step):
        """Saves parameters and optimizer state_dict as a checkpoint.

        Args:
            checkpoint_step (int): iteration to label checkpoint with
        """
        optimizer_state_dict = self._optimizer.state_dict()
        torch.save(
            dict(meta_parameters=self._meta_parameters,
                 inner_lrs=self._inner_lrs,
                 optimizer_state_dict=optimizer_state_dict),
            f'{os.path.join(self._log_dir, "state")}{checkpoint_step}.pt'
        )
        print('Saved checkpoint.')

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

def main(args):
    if args.device == "gpu" and torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    print("Using device: ", DEVICE)
    random.seed(0)

    log_dir = args.log_dir
    if log_dir is None:
        log_dir = f'./logs/model_{args.model}/normalized_way_{args.num_way}.support_{args.num_support}.query_{args.num_query}.inner_steps_{args.num_inner_steps}.inner_lr_{args.inner_lr}.learn_inner_lrs_{args.learn_inner_lrs}.outer_lr_{args.outer_lr}.batch_size_{args.meta_batch_size}.train_iter_{args.meta_train_iterations}.hd_{NUM_HIDDEN_CHANNELS}.cvl_{NUM_CONV_LAYERS}'  # pylint: disable=line-too-long
    print(f'log_dir: {log_dir}')
    writer = tensorboard.SummaryWriter(log_dir=log_dir)

    if args.model == 'maml':
        maml = MAML(
        args.num_way,
        args.num_inner_steps,
        args.inner_lr,
        args.learn_inner_lrs,
        args.outer_lr,
        log_dir,
        DEVICE
    )
    else:
        print(f"ERROR: Model '{args.model}' is not implemented yet")
        exit()

    if args.checkpoint_step > -1:
        maml.load(args.checkpoint_step)
    else:
        print('Checkpoint loading skipped.')



    if not args.test:

        meta_train_iterable = DataGenerator(
        args.num_videos,
        args.num_support + args.num_query,
        batch_type="train",
        cache=False,
        generate_new_tasks=True
        )
        meta_train_loader = iter(
            torch.utils.data.DataLoader(
                meta_train_iterable,
                batch_size=args.meta_batch_size,
                num_workers=args.num_workers,
                pin_memory=True,
            )
        )
        # for i in range(args.meta_batch_size):
        #     images, labels, order = next(meta_train_loader)
        #     support_frames, support_labels, query_frames, query_labels = get_support_query(images, labels)
        #     print_support_query_shapes(support_frames, support_labels, query_frames, query_labels)


        # for batch_idx in range(query_labels.shape[0]):
        #     for channel_idx in range(query_labels.shape[1]):
        #         for row_idx in range(query_labels.shape[2]):
        #             for time_idx in range(query_labels.shape[3]):
        #                 for element_idx in range(query_labels.shape[4]):
        #                     label_value = query_labels[batch_idx, channel_idx, row_idx, time_idx, element_idx]
        #                     print(label_value.item())



        meta_val_iterable = DataGenerator(
        args.num_videos,
        args.num_support + args.num_query,
        batch_type="val",
        cache=False,
        generate_new_tasks=True
        )
        meta_val_loader = iter(
            torch.utils.data.DataLoader(
                meta_val_iterable,
                batch_size=args.meta_batch_size,
                num_workers=args.num_workers,
                pin_memory=True,
            )
        )

    print("Loaded data")
    if args.model == 'maml':
        maml.train(
            meta_train_loader,
            meta_val_loader,
            writer
        )
    else:
        pass
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default=None, help='directory to save to or load from')
    parser.add_argument('--model', type=str, default='maml', help='model to run')
    parser.add_argument('--num_videos', type=int, default=1, help='number of videos to include in the support set')
    parser.add_argument('--num_way', type=int, default=23, help='number of classes in a task')
    parser.add_argument('--num_support', type=int, default=3, help='number of support examples per class in a task')
    parser.add_argument('--num_query', type=int, default=1, help='number of query examples per class in a task')
    parser.add_argument('--meta_batch_size', type=int, default=2, help='number of tasks per outer-loop update')
    parser.add_argument('--meta_train_iterations', type=int, default=200, help='number of baches of tasks to iterate through for train')
    parser.add_argument('--meta_val_iterations', type=int, default=50, help='number of baches of tasks to iterate through for val per every check')
    parser.add_argument('--num_inner_steps', type=int, default=1, help='number of inner-loop updates')
    parser.add_argument('--inner_lr', type=float, default=0.4, help='inner-loop learning rate initialization')
    parser.add_argument('--learn_inner_lrs', default=False, action='store_true', help='whether to optimize inner-loop learning rates')
    parser.add_argument('--outer_lr', type=float, default=0.001, help='outer-loop learning rate')
    parser.add_argument('--test', default=False, action='store_true', help='train or test')
    parser.add_argument('--num_workers', type=int, default=int(multiprocessing.cpu_count()/2), help=('needed to specify dataloader'))
    parser.add_argument('--checkpoint_step', type=int, default=-1, help=('checkpoint iteration to load for resuming training, or for evaluation (-1 is ignored)'))
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--device', type=str, default='gpu')
    args = parser.parse_args()
    main(args)



        # images, labels, random_order = next(meta_train_loader)
        # support_frames = images[:,:-args.num_query]
        # support_labels = labels[:,:-args.num_query]
        # query_frames = images[:,-args.num_query:]
        # query_labels = labels[:,-args.num_query:]
        # support_frames = torch.squeeze(support_frames, dim=2)
        # support_labels = torch.squeeze(support_labels, dim=2)
        # query_frames = torch.squeeze(query_frames, dim=2)
        # query_labels = torch.squeeze(query_labels, dim=2)
        # print("\nSupport Set Shape:", support_frames.shape)
        # print("Support Label Shape:", support_labels.shape)
        # print("Query Set Shape:", query_frames.shape)
        # print("Query Label Shape:", query_labels.shape)
        # print("random order:", random_order)