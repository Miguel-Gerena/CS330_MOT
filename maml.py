"""Implementation of model-agnostic meta-learning for Sportsmot."""
import sys
sys.path.append('..')
sys.path.append('/home/LavinuxCS330azureuser/CS330_MOT_Project_V2/CS330_MOT/Stark')
from tqdm import tqdm

import argparse
import os
import numpy as np
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch import nn
import torch.nn.functional as F
from torch import autograd
from torch.utils import tensorboard
from lib.models.stark.stark_st import build_starkst
import importlib
from data.load_data import DataGenerator
import copy
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict
from lib.train.trainers import LTRTrainer
NUM_INPUT_CHANNELS = 1
NUM_HIDDEN_CHANNELS = 32
KERNEL_SIZE = 3
NUM_CONV_LAYERS = 4
SUMMARY_INTERVAL = 10
SAVE_INTERVAL = 100
LOG_INTERVAL = 10
VAL_INTERVAL = LOG_INTERVAL * 5
NUM_TEST_TASKS = 600

def score(logits, labels):
    """Returns the mean accuracy of a model's predictions on a set of examples.

    Args:
        logits (torch.Tensor): model predicted logits
            shape (examples, classes)
        labels (torch.Tensor): classification labels from 0 to num_classes - 1
            shape (examples,)
    """

    assert logits.dim() == 2
    assert labels.dim() == 1
    assert logits.shape[0] == labels.shape[0]
    y = torch.argmax(logits, dim=-1) == labels
    y = y.type(torch.float)
    return torch.mean(y).item()

class MAML:
    """Trains and assesses a MAML."""

    def __init__(
            self,
            tracking_model,
            num_outputs,
            num_inner_steps,
            inner_lr,
            learn_inner_lrs,
            outer_lr,
            log_dir,
            device
    ):
        """Inits MAML.

        The network consists of four convolutional blocks followed by a linear
        head layer. Each convolutional block comprises a convolution layer, a
        batch normalization layer, and ReLU activation.

        Note that unlike conventional use, batch normalization is always done
        with batch statistics, regardless of whether we are training or
        evaluating. This technically makes meta-learning transductive, as
        opposed to inductive.

        Args:
            num_outputs (int): dimensionality of output, i.e. number of classes
                in a task
            num_inner_steps (int): number of inner-loop optimization steps
            inner_lr (float): learning rate for inner-loop optimization
                If learn_inner_lrs=True, inner_lr serves as the initialization
                of the learning rates.
            learn_inner_lrs (bool): whether to learn the above
            outer_lr (float): learning rate for outer-loop optimization
            log_dir (str): path to logging directory
            device (str): device to be used
        """
        meta_parameters = {}

        self.device = device
        self.tracking_model = tracking_model

        config_module = importlib.import_module("lib.config.stark_st1.config")
        cfg = config_module.cfg
        net = build_starkst(cfg)

        # Load pretrained weights from the Stark model
        stark_pretrained_weights = torch.load("/home/LavinuxCS330azureuser/CS330_MOT_Project_V2/CS330_MOT/Stark/checkpoints/train/stark_st2/baseline/STARKST_ep0050.pth.tar")

        # Initialize meta_parameters with pretrained weights relevant to your model's structure
        meta_parameters = {}

        net.load_state_dict(stark_pretrained_weights["net"])
        self.net = net

        for key, value in stark_pretrained_weights["net"].items():
            # if key.startswith('backbone') or key.startswith('transformer') or key.startswith('box_head') or key.startswith('cls_head'):
            meta_parameters[key] = value

        self._meta_parameters = meta_parameters

        self._num_inner_steps = num_inner_steps
        self._inner_lrs = {
            k: torch.tensor(inner_lr, requires_grad=learn_inner_lrs)
            for k in self._meta_parameters.keys()
        }
        self._outer_lr = outer_lr

        self._optimizer = torch.optim.Adam(
            list(self._meta_parameters.values()) +
            list(self._inner_lrs.values()),
            lr=self._outer_lr
        )
        self._log_dir = log_dir
        os.makedirs(self._log_dir, exist_ok=True)
        settings = importlib.import_module('lib.train.admin.local').EnvironmentSettings()

        self._start_train_step = 0

    def _forward(self, images, parameters):
        """Computes predicted outputs using the pre-trained STARK model.

            Args:
            images (Tensor): batch of input images
                shape (num_images, channels, height, width)
            parameters (dict[str, Tensor]): parameters (not used in STARK forward pass)

            Returns:
                outputs: the output from the STARK model
            """
        # Ensure images are in the correct device
        images = images.to(self.device)
        
        # Pass images through the STARK model (net)
        outputs = self.net(images)

        #TODO check how outpu looks like, output the predictions on the query examples

    def _inner_loop(self, images, labels, train):
        accuracies = []
        parameters = {
            k: torch.clone(v).requires_grad_(v.dtype.is_floating_point)
            for k, v in self._meta_parameters.items()
        }   

        # Create a temporary module for the inner loop
        temp_model = self.net
        temp_model.load_state_dict(parameters, strict=False)
        temp_model.train(train)

        inner_optimizer = torch.optim.Adam(temp_model.parameters(), lr=0.0001) #inner learnin rate todo change

        settings = importlib.import_module('lib.train.admin.local').EnvironmentSettings()

        for step in range(self._num_inner_steps):

            # Set up the trainer for the temporary model
            trainer = LTRTrainer(
                actor=temp_model, 
                loaders=[images], 
                optimizer=inner_optimizer ,
                settings=settings, 
                lr_scheduler=None, 
                use_amp=False
            )

            # Run the training cycle with the current inner loop step's data
            trainer.cycle_dataset(images)

        # Calculate and record the accuracy
        accuracy = score(temp_model(images), labels) #TODO 
        accuracies.append(accuracy)

        return parameters, accuracies

   

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

        for task_idx in range(args.batch_size):  # Iterate over tasks in the batch
            # print("task num:", task_idx+1)
            images_task = images[task_idx]  # Images for the current task
            labels_task = labels[task_idx]  # Labels for the current task

            images_support = images_task[:args.num_support].to(self.device) 
            labels_support = labels_task[:args.num_support].to(self.device) 
            images_query = images_task[args.num_support:].to(self.device)  
            labels_query = labels_task[args.num_support:].to(self.device)   

            adapted_params, accuracies_support, _ = self._inner_loop(images_support, labels_support, train) 
            logits_query = self._forward(images_query, adapted_params)
            
            labels = data['label'].view(-1)  # (batch, ) 0 or 1
            loss = self.loss_weight["cls"] * self.objective['cls'](pred_dict["pred_logits"].view(-1), labels)


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

        # print(f"Pre-adaptation accuracies: {pre_adapt_accuracy_mean}")
        # print(f"Post-adaptation accuracies: {post_adapt_accuracy_mean}")



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

        # print(f" query accuracies: {accuracy_query_mean}")

        return outer_loss, pre_adapt_accuracy_mean, post_adapt_accuracy_mean, accuracy_query_mean
    
    def maml_train(self, dataloader_meta_train, dataloader_meta_val, writer):
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

            for i_step, (images, labels, sports_order) in tqdm(enumerate(dataloader_meta_train, start=self._start_train_step), total=args.num_train_iterations):
                if i_step >= args.num_train_iterations:
                    break
                # print(f"Train Iteration: {i_step + 1}/{args.num_train_iterations}")

                # if (i_step + 1) % 2 == 0:
                self._optimizer.step()
                self._optimizer.zero_grad()
                outer_loss, pre_adapt_accuracy_mean, post_adapt_accuracy_mean, accuracy_query = (
                    self._outer_step(images, labels, train=True)
                )
                # outer_loss /= 2
                outer_loss.backward()
                self._optimizer.step()

                

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
                    # print("\nIn val interval check")
                    losses = []
                    val_accuracies_pre_adapt_support = {}
                    val_accuracies_post_adapt_support = {}
                    val_accuracies_post_adapt_query = {}

                    # Iterate through validation batches
                    for batch_idx, val_batch in enumerate(dataloader_meta_val):
                        if batch_idx >= args.meta_val_iterations:
                            break
                        # print(f"Batch {batch_idx + 1}/{args.meta_val_iterations}")
                        val_images, val_labels, _ = val_batch  # Unpack the batch

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

                    # print("\n\nDONE WITH CHECK")
                    

                if i_step % SAVE_INTERVAL == 0:
                    print(torch.cuda.memory_summary())
                    self._save(i_step)




    # def train(self, dataloader_meta_train, dataloader_meta_val, writer):
    #     """Train the MAML.

    #     Consumes dataloader_meta_train to optimize MAML meta-parameters
    #     while periodically validating on dataloader_meta_val, logging metrics, and
    #     saving checkpoints.

    #     Args:
    #         dataloader_meta_train (DataLoader): loader for train tasks
    #         dataloader_meta_val (DataLoader): loader for validation tasks
    #         writer (SummaryWriter): TensorBoard logger
    #     """
    #     print(f'Starting training at iteration {self._start_train_step}.')
    #     for i_step, task_batch in enumerate(
    #             dataloader_meta_train,
    #             start=self._start_train_step
    #     ):
    #         self._optimizer.zero_grad()
    #         outer_loss, accuracies_support, accuracy_query = (
    #             self._outer_step(task_batch, train=True)
    #         )
    #         outer_loss.backward()
    #         self._optimizer.step()

    #         if i_step % LOG_INTERVAL == 0:
    #             print(
    #                 f'Iteration {i_step}: '
    #                 f'loss: {outer_loss.item():.3f}, '
    #                 f'pre-adaptation support accuracy: '
    #                 f'{accuracies_support[0]:.3f}, '
    #                 f'post-adaptation support accuracy: '
    #                 f'{accuracies_support[-1]:.3f}, '
    #                 f'post-adaptation query accuracy: '
    #                 f'{accuracy_query:.3f}'
    #             )
    #             writer.add_scalar('loss/train', outer_loss.item(), i_step)
    #             writer.add_scalar(
    #                 'train_accuracy/pre_adapt_support',
    #                 accuracies_support[0],
    #                 i_step
    #             )
    #             writer.add_scalar(
    #                 'train_accuracy/post_adapt_support',
    #                 accuracies_support[-1],
    #                 i_step
    #             )
    #             writer.add_scalar(
    #                 'train_accuracy/post_adapt_query',
    #                 accuracy_query,
    #                 i_step
    #             )

    #         if i_step % VAL_INTERVAL == 0:
    #             losses = []
    #             accuracies_pre_adapt_support = []
    #             accuracies_post_adapt_support = []
    #             accuracies_post_adapt_query = []
    #             for val_task_batch in dataloader_meta_val:
    #                 outer_loss, accuracies_support, accuracy_query = (
    #                     self._outer_step(val_task_batch, train=False)
    #                 )
    #                 losses.append(outer_loss.item())
    #                 accuracies_pre_adapt_support.append(accuracies_support[0])
    #                 accuracies_post_adapt_support.append(accuracies_support[-1])
    #                 accuracies_post_adapt_query.append(accuracy_query)
    #             loss = np.mean(losses)
    #             accuracy_pre_adapt_support = np.mean(
    #                 accuracies_pre_adapt_support
    #             )
    #             accuracy_post_adapt_support = np.mean(
    #                 accuracies_post_adapt_support
    #             )
    #             accuracy_post_adapt_query = np.mean(
    #                 accuracies_post_adapt_query
    #             )
    #             print(
    #                 f'Validation: '
    #                 f'loss: {loss:.3f}, '
    #                 f'pre-adaptation support accuracy: '
    #                 f'{accuracy_pre_adapt_support:.3f}, '
    #                 f'post-adaptation support accuracy: '
    #                 f'{accuracy_post_adapt_support:.3f}, '
    #                 f'post-adaptation query accuracy: '
    #                 f'{accuracy_post_adapt_query:.3f}'
    #             )
    #             writer.add_scalar('loss/val', loss, i_step)
    #             writer.add_scalar(
    #                 'val_accuracy/pre_adapt_support',
    #                 accuracy_pre_adapt_support,
    #                 i_step
    #             )
    #             writer.add_scalar(
    #                 'val_accuracy/post_adapt_support',
    #                 accuracy_post_adapt_support,
    #                 i_step
    #             )
    #             writer.add_scalar(
    #                 'val_accuracy/post_adapt_query',
    #                 accuracy_post_adapt_query,
    #                 i_step
    #             )

    #         if i_step % SAVE_INTERVAL == 0:
    #             self._save(i_step)

    def test(self, dataloader_test):
        """Evaluate the MAML on test tasks.

        Args:
            dataloader_test (DataLoader): loader for test tasks
        """
        accuracies = []
        for task_batch in dataloader_test:
            _, _, accuracy_query = self._outer_step(task_batch, train=False)
            accuracies.append(accuracy_query)
        mean = np.mean(accuracies)
        std = np.std(accuracies)
        mean_95_confidence_interval = 1.96 * std / np.sqrt(NUM_TEST_TASKS)
        print(
            f'Accuracy over {NUM_TEST_TASKS} test tasks: '
            f'mean {mean:.3f}, '
            f'95% confidence interval {mean_95_confidence_interval:.3f}'
        )

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


def main(args):

    print(args)

    if args.device == "gpu" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # on MPS the derivative for aten::linear_backward is not implemented ... Waiting for PyTorch 2.1.0
        DEVICE = "mps"

        # Due to the above, default for now to cpu
         #DEVICE = "cpu"
    elif args.device == "gpu" and torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    print("Using device: ", DEVICE)

    log_dir = args.log_dir
    if log_dir is None:
        log_dir = f'./logs/maml/omniglot.way_{args.num_way}.support_{args.num_support}.query_{args.num_query}.inner_steps_{args.num_inner_steps}.inner_lr_{args.inner_lr}.learn_inner_lrs_{args.learn_inner_lrs}.outer_lr_{args.outer_lr}.batch_size_{args.batch_size}'  # pylint: disable=line-too-long
    print(f'log_dir: {log_dir}')
    writer = tensorboard.SummaryWriter(log_dir=log_dir)

    maml = MAML(
        args.tracking_model,
        args.num_way,
        args.num_inner_steps,
        args.inner_lr,
        args.learn_inner_lrs,
        args.outer_lr,
        log_dir,
        DEVICE
    )

    if args.checkpoint_step > -1:
        maml.load(args.checkpoint_step)
    else:
        print('Checkpoint loading skipped.')

    if not args.test:
        num_training_tasks = args.batch_size * (args.num_train_iterations -
                                                args.checkpoint_step - 1)

        print("Loading training data.. ")
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
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=True,
            )
        )

        print("Loading validation data.. ")
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
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=True,
            )
        )

        print(f'Training on {num_training_tasks} tasks with composition: '
                    f'num_way={args.num_way}, '
                    f'num_support={args.num_support}, '
                    f'num_query={args.num_query}'
                )

        maml.maml_train(
            meta_train_loader,
            meta_val_loader,
            writer
        )
    else:
        print(
            f'Testing on tasks with composition '
            f'num_way={args.num_way}, '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}'
        )
        dataloader_test = omniglot.get_omniglot_dataloader(
            'test',
            1,
            args.num_way,
            args.num_support,
            args.num_query,
            NUM_TEST_TASKS,
            args.num_workers
        )
        maml.test(dataloader_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a MAML!')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='directory to save to or load from')
    parser.add_argument('--num_way', type=int, default=5,
                        help='number of classes in a task')
    parser.add_argument('--num_support', type=int, default=1,
                        help='number of support examples per class in a task')
    parser.add_argument('--num_videos', type=int, default=1,
                         help='number of videos to include in the support set')
    parser.add_argument('--num_query', type=int, default=15,
                        help='number of query examples per class in a task')
    parser.add_argument('--num_inner_steps', type=int, default=1,
                        help='number of inner-loop updates')
    parser.add_argument('--inner_lr', type=float, default=0.4,
                        help='inner-loop learning rate initialization')
    parser.add_argument('--learn_inner_lrs', default=False, action='store_true',
                        help='whether to optimize inner-loop learning rates')
    parser.add_argument('--outer_lr', type=float, default=0.001,
                        help='outer-loop learning rate')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='number of tasks per outer-loop update')
    parser.add_argument('--num_train_iterations', type=int, default=15000,
                        help='number of outer-loop updates to train for')
    parser.add_argument('--test', default=False, action='store_true',
                        help='train or test')
    parser.add_argument('--checkpoint_step', type=int, default=-1,
                        help=('checkpoint iteration to load for resuming '
                              'training, or for evaluation (-1 is ignored)'))
    parser.add_argument('--num_workers', type=int, default=2, 
                        help=('needed to specify omniglot dataloader'))
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--tracking_model', type=str, default='stark', help="maml_hw2_original if using code from hw2, else stark")
    parser.add_argument('--script', type=str, default='stark', help="maml_hw2_original if using code from hw2, else stark")

    args = parser.parse_args()

    if args.cache == True:
        # Download Omniglot Dataset
        if not os.path.isdir("./omniglot_resized"):
            gdd.download_file_from_google_drive(
                file_id="1iaSFXIYC3AB8q9K_M-oVMa4pmB7yKMtI",
                dest_path="./omniglot_resized.zip",
                unzip=True,
            )
        assert os.path.isdir("./omniglot_resized")
    else:
        main(args)



    # python tracking/train.py --script stark_lightning_X_trt --config baseline_rephead_4_lite_search5 --save_dir . --mode multiple --nproc_per_node 8




