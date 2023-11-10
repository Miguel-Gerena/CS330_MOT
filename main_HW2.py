import sys
sys.path.append('..')
import argparse
import os
import random
import json
import numpy as np
import torch

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from torch import nn
import torch.nn.functional as F
from torch import autograd
from torch.utils import tensorboard
# sys.path.append('./')
# import maml
sys.path.append('./ak_util/')
import load_by_sport


NUM_INPUT_CHANNELS = 1
NUM_HIDDEN_CHANNELS = 32
KERNEL_SIZE = 3
NUM_CONV_LAYERS = 4
SUMMARY_INTERVAL = 10
SAVE_INTERVAL = 100
LOG_INTERVAL = 10
VAL_INTERVAL = LOG_INTERVAL * 5
NUM_TEST_TASKS = 600

class MAML:
    """Trains and assesses a MAML."""

    def __init__(
            self,
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

        # construct feature extractor
        in_channels = NUM_INPUT_CHANNELS
        for i in range(NUM_CONV_LAYERS):
            meta_parameters[f'conv{i}'] = nn.init.xavier_uniform_(
                torch.empty(
                    NUM_HIDDEN_CHANNELS,
                    in_channels,
                    KERNEL_SIZE,
                    KERNEL_SIZE,
                    requires_grad=True,
                    device=self.device
                )
            )
            meta_parameters[f'b{i}'] = nn.init.zeros_(
                torch.empty(
                    NUM_HIDDEN_CHANNELS,
                    requires_grad=True,
                    device=self.device
                )
            )
            in_channels = NUM_HIDDEN_CHANNELS

        # construct linear head layer
        meta_parameters[f'w{NUM_CONV_LAYERS}'] = nn.init.xavier_uniform_(
            torch.empty(
                num_outputs,
                NUM_HIDDEN_CHANNELS,
                requires_grad=True,
                device=self.device
            )
        )
        meta_parameters[f'b{NUM_CONV_LAYERS}'] = nn.init.zeros_(
            torch.empty(
                num_outputs,
                requires_grad=True,
                device=self.device
            )
        )

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

        self._start_train_step = 0

    def _forward(self, images, parameters):
        """Computes predicted classification logits.

        Args:
            images (Tensor): batch of Omniglot images
                shape (num_images, channels, height, width)
            parameters (dict[str, Tensor]): parameters to use for
                the computation

        Returns:
            a Tensor consisting of a batch of logits
                shape (num_images, classes)
        """
        x = images
        for i in range(NUM_CONV_LAYERS):
            x = F.conv2d(
                input=x,
                weight=parameters[f'conv{i}'],
                bias=parameters[f'b{i}'],
                stride=1,
                padding='same'
            )
            x = F.batch_norm(x, None, None, training=True)
            x = F.relu(x)
        x = torch.mean(x, dim=[2, 3])
        return F.linear(
            input=x,
            weight=parameters[f'w{NUM_CONV_LAYERS}'],
            bias=parameters[f'b{NUM_CONV_LAYERS}']
        )

    def _inner_loop(self, images, labels, train):
        """Computes the adapted network parameters via the MAML inner loop.

        Args:
            images (Tensor): task support set inputs
                shape (num_images, channels, height, width)
            labels (Tensor): task support set outputs
                shape (num_images,)
            train (bool): whether we are training or evaluating

        Returns:
            parameters (dict[str, Tensor]): adapted network parameters
            accuracies (list[float]): support set accuracy over the course of
                the inner loop, length num_inner_steps + 1
            gradients(list[float]): gradients computed from auto.grad, just needed
                for autograders, no need to use this value in your code and feel to replace
                with underscore       
        """
        accuracies = []
        parameters = {
            k: torch.clone(v)
            for k, v in self._meta_parameters.items()
        }
        gradients = None
        num_inner_steps = self._num_inner_steps
        
        for step in range(num_inner_steps):
            logits_1 = self._forward(images, parameters)
            loss = F.cross_entropy(logits_1, labels)
            accuracy_1 = util.score(logits_1, labels)
            accuracies.append(accuracy_1)
            # Calculate gradients using autograd.grad
            if train: 
                gradients = torch.autograd.grad(loss, parameters.values(), create_graph=True)
            else:
                gradients = torch.autograd.grad(loss, parameters.values(), create_graph=False)

            # Update parameters using gradient descent with individual inner learning rates
            for idx, key in enumerate(parameters.keys()):
                parameters[key] = parameters[key] - self._inner_lrs[key] * gradients[idx] 

        logits_2 = self._forward(images, parameters)
        accuracy_2 = util.score(logits_2, labels)
        accuracies.append(accuracy_2)     

        return parameters, accuracies, gradients

    def _outer_step(self, task_batch, train):
        """Computes the MAML loss and metrics on a batch of tasks.

        Args:
            task_batch (tuple): batch of tasks from an Omniglot DataLoader
            train (bool): whether we are training or evaluating

        Returns:
            outer_loss (Tensor): mean MAML loss over the batch, scalar
            accuracies_support (ndarray): support set accuracy over the
                course of the inner loop, averaged over the task batch
                shape (num_inner_steps + 1,)
            accuracy_query (float): query set accuracy of the adapted
                parameters, averaged over the task batch
        """
        outer_loss_batch = []
        accuracies_support_batch = []
        accuracy_query_batch = []
        for task in task_batch:
            print(len(task))
            exit()
            images_support, labels_support, images_query, labels_query = task
            images_support = images_support.to(self.device)
            labels_support = labels_support.to(self.device)
            images_query = images_query.to(self.device)
            labels_query = labels_query.to(self.device)

            adapted_params, accuracies_support, _ = self._inner_loop(images_support, labels_support, train)
            logits_query = self._forward(images_query, adapted_params)
            loss_query = F.cross_entropy(logits_query, labels_query)
            accuracy_query = util.score(logits_query, labels_query)

            outer_loss_batch.append(loss_query)
            accuracies_support_batch.append(accuracies_support)
            accuracy_query_batch.append(accuracy_query)

        outer_loss = torch.mean(torch.stack(outer_loss_batch))
        accuracies_support = np.mean(
            accuracies_support_batch,
            axis=0
        )
        accuracy_query = np.mean(accuracy_query_batch)
        return outer_loss, accuracies_support, accuracy_query

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
        print(f"entire batch length: {len(dataloader_meta_train)}")
        for i_step, task_batch in enumerate(
                dataloader_meta_train,
                start=self._start_train_step
        ):
            print(f"Iteration: {i_step}")
            print(f"Task batch structure: {[type(item) for item in task_batch]}")
            print(f"Task batch length: {len(task_batch)}")
            self._optimizer.zero_grad()
            outer_loss, accuracies_support, accuracy_query = (
                self._outer_step(task_batch, train=True)
            )
            outer_loss.backward()
            self._optimizer.step()

            if i_step % LOG_INTERVAL == 0:
                print(
                    f'Iteration {i_step}: '
                    f'loss: {outer_loss.item():.3f}, '
                    f'pre-adaptation support accuracy: '
                    f'{accuracies_support[0]:.3f}, '
                    f'post-adaptation support accuracy: '
                    f'{accuracies_support[-1]:.3f}, '
                    f'post-adaptation query accuracy: '
                    f'{accuracy_query:.3f}'
                )
                writer.add_scalar('loss/train', outer_loss.item(), i_step)
                writer.add_scalar(
                    'train_accuracy/pre_adapt_support',
                    accuracies_support[0],
                    i_step
                )
                writer.add_scalar(
                    'train_accuracy/post_adapt_support',
                    accuracies_support[-1],
                    i_step
                )
                writer.add_scalar(
                    'train_accuracy/post_adapt_query',
                    accuracy_query,
                    i_step
                )

            if i_step % VAL_INTERVAL == 0:
                losses = []
                accuracies_pre_adapt_support = []
                accuracies_post_adapt_support = []
                accuracies_post_adapt_query = []
                for val_task_batch in dataloader_meta_val:
                    outer_loss, accuracies_support, accuracy_query = (
                        self._outer_step(val_task_batch, train=False)
                    )
                    losses.append(outer_loss.item())
                    accuracies_pre_adapt_support.append(accuracies_support[0])
                    accuracies_post_adapt_support.append(accuracies_support[-1])
                    accuracies_post_adapt_query.append(accuracy_query)
                loss = np.mean(losses)
                accuracy_pre_adapt_support = np.mean(
                    accuracies_pre_adapt_support
                )
                accuracy_post_adapt_support = np.mean(
                    accuracies_post_adapt_support
                )
                accuracy_post_adapt_query = np.mean(
                    accuracies_post_adapt_query
                )
                print(
                    f'Validation: '
                    f'loss: {loss:.3f}, '
                    f'pre-adaptation support accuracy: '
                    f'{accuracy_pre_adapt_support:.3f}, '
                    f'post-adaptation support accuracy: '
                    f'{accuracy_post_adapt_support:.3f}, '
                    f'post-adaptation query accuracy: '
                    f'{accuracy_post_adapt_query:.3f}'
                )
                writer.add_scalar('loss/val', loss, i_step)
                writer.add_scalar(
                    'val_accuracy/pre_adapt_support',
                    accuracy_pre_adapt_support,
                    i_step
                )
                writer.add_scalar(
                    'val_accuracy/post_adapt_support',
                    accuracy_post_adapt_support,
                    i_step
                )
                writer.add_scalar(
                    'val_accuracy/post_adapt_query',
                    accuracy_post_adapt_query,
                    i_step
                )

            if i_step % SAVE_INTERVAL == 0:
                self._save(i_step)

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
















TRAIN_COUNTS_JSON_PATH = './train_counts.json'
VAL_COUNTS_JSON_PATH = './val_counts.json'
TEST_COUNTS_JSON_PATH = './test_counts.json'

TRAIN_DATASET_PATH = './dataset/train/'
VAL_DATASET_PATH = './dataset/val/'
TEST_DATASET_PATH = './dataset/test/'

# NUM_INPUT_CHANNELS = 1
# NUM_HIDDEN_CHANNELS = 32
# KERNEL_SIZE = 3
# NUM_CONV_LAYERS = 4
# SUMMARY_INTERVAL = 10
# SAVE_INTERVAL = 100
# LOG_INTERVAL = 10
# VAL_INTERVAL = LOG_INTERVAL * 5
# NUM_TEST_TASKS = 50

def main(args):
    if args.device == "gpu" and torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    print("Using device: ", DEVICE)
    random.seed(0)
    strategy_function = getattr(load_by_sport, args.frame_selection_strategy, None)
    if strategy_function is None:
        raise ValueError(f"Invalid frame selection strategy: {args.frame_selection_strategy}")


    log_dir = args.log_dir
    if log_dir is None:
        log_dir = f'./logs/model_{args.model}/way_{args.num_way}.support_{args.num_support}.query_{args.num_query}.inner_steps_{args.num_inner_steps}.inner_lr_{args.inner_lr}.learn_inner_lrs_{args.learn_inner_lrs}.outer_lr_{args.outer_lr}.batch_size_{args.batch_size}'  # pylint: disable=line-too-long
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
        num_training_tasks = args.batch_size * (args.num_train_iterations - args.checkpoint_step - 1)
        # Load the train dictionary
        train_sports_dict = load_by_sport.get_videos_file_path_dict(TRAIN_COUNTS_JSON_PATH, TRAIN_DATASET_PATH)

        print(f"loading meta train num batches: {args.batch_size}\nWith num tasks per batch: {num_training_tasks}")
        train_tasks = load_by_sport.get_batches(train_sports_dict, args.batch_size, args.num_support_videos, args.num_query_videos, args.num_support, args.num_query, num_training_tasks, DEVICE, strategy_function)


        print(f"Done\nloading meta validation num batches: {args.batch_size}\nWith num tasks per batch: {args.batch_size * 4}")
        val_sports_dict = load_by_sport.get_videos_file_path_dict(VAL_COUNTS_JSON_PATH, VAL_DATASET_PATH)
        val_tasks = load_by_sport.get_batches(val_sports_dict, args.batch_size, args.num_support_videos, args.num_query_videos, args.num_support, args.num_query, args.batch_size * 4, DEVICE, strategy_function)
        print(f"Done")

        train_tasks = [(support_frames.to(DEVICE), support_labels.to(DEVICE), query_frames.to(DEVICE), query_labels.to(DEVICE)) for support_frames, support_labels, query_frames, query_labels in train_tasks]
        val_tasks = [(support_frames.to(DEVICE), support_labels.to(DEVICE), query_frames.to(DEVICE), query_labels.to(DEVICE)) for support_frames, support_labels, query_frames, query_labels in val_tasks]

    if args.model == 'maml':
        maml.train(
            train_tasks,
            val_tasks,
            writer
        )

    else:
        print(f"loading meta test num batches: 1\nWith num tasks per batch: {NUM_TEST_TASKS}")
        test_sports_dict = load_by_sport.get_videos_file_path_dict(TEST_COUNTS_JSON_PATH, TEST_DATASET_PATH)
        test_tasks = load_by_sport.get_batches(test_sports_dict, 1, args.num_support_videos, args.num_query_videos, args.num_support, args.num_query, NUM_TEST_TASKS, DEVICE, strategy_function)
        print(f"Done\nNow in the testing process..")
        maml.test(test_tasks)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default=None,
                        help='directory to save to or load from')
    parser.add_argument('--model', type=str, default='maml',
                        help='model to run')
    parser.add_argument('--frame_selection_strategy', type=str, default="consecutive_frame_selection",
                        help='The function in load_by_sport file that we want to use for splitting the support/query sets. EX: consecutive vs random')
    parser.add_argument('--num_support_videos', type=int, default=2,
                        help='number of videos to include in the support set')
    parser.add_argument('--num_query_videos', type=int, default=2,
                        help='number of videos to include in the query set')
    parser.add_argument('--num_way', type=int, default=5,
                        help='number of classes in a task')
    parser.add_argument('--num_support', type=int, default=5,
                        help='number of support examples per class in a task')
    parser.add_argument('--num_query', type=int, default=15,
                        help='number of query examples per class in a task')
    parser.add_argument('--num_tasks_per_batch', type=int, default=3,
                        help='number of inner-loop updates')
    parser.add_argument('--num_inner_steps', type=int, default=1,
                        help='number of inner-loop updates')
    parser.add_argument('--inner_lr', type=float, default=0.4,
                        help='inner-loop learning rate initialization')
    parser.add_argument('--learn_inner_lrs', default=False, action='store_true',
                        help='whether to optimize inner-loop learning rates')
    parser.add_argument('--outer_lr', type=float, default=0.001,
                        help='outer-loop learning rate')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='number of tasks per outer-loop update')
    parser.add_argument('--num_train_iterations', type=int, default=15000,
                        help='number of outer-loop updates to train for')
    parser.add_argument('--test', default=False, action='store_true',
                        help='train or test')
    parser.add_argument('--checkpoint_step', type=int, default=-1,
                        help=('checkpoint iteration to load for resuming '
                              'training, or for evaluation (-1 is ignored)'))
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--device', type=str, default='gpu')
    args = parser.parse_args()
    main(args)