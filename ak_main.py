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
        # 6 because of the 6 outputs
        for output_component in range(6):
            meta_parameters[f'w{NUM_CONV_LAYERS}_{output_component}'] = nn.init.xavier_uniform_(
                torch.empty(
                    num_outputs,
                    NUM_HIDDEN_CHANNELS,
                    requires_grad=True,
                    device=self.device
                )
            )
            meta_parameters[f'b{NUM_CONV_LAYERS}_{output_component}'] = nn.init.zeros_(
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
        """Computes predicted outputs for frame id, player id, bb_left, bb_top, bb_width, and bb_height.

        Args:
            images (Tensor): batch of images
                shape (num_images, channels, height, width)
            parameters (dict[str, Tensor]): parameters to use for the computation

        Returns:
            a tuple of Tensors consisting of batches of logits for each component
            shape (batch_size, num_videos, num_sports, num_players, num_components)
        """
        x = images
        for i in range(NUM_CONV_LAYERS):
            x = F.conv2d(
                input=x,
                weight=parameters[f'conv{i}'],
                bias=parameters[f'b{i}'],
                stride=2,
                padding=0
            )
            x = F.batch_norm(x, None, None, training=True)
            x = F.relu(x)
        x = torch.mean(x, dim=[-1, -2])

        # Initialize lists to store the outputs for each component
        outputs = []

        # Iterate through each output component (frame id, player id, bb_left, bb_top, bb_width, bb_height)
        for component in range(6):
            # Compute the output for the current component
            component_output = F.linear(input=x, weight=parameters[f'w{NUM_CONV_LAYERS}_{component}'], bias=parameters[f'b{NUM_CONV_LAYERS}_{component}'])
            
            # Reshape the output to match the shape of the labels
            component_output = component_output.view(images.shape[0], -1, component_output.shape[-1]).squeeze(dim=1)

            outputs.append(component_output)

        # Return the tuple of outputs for all components
        return tuple(outputs)


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
        accuracy_dict = {}
        parameters = {
            k: torch.clone(v)
            for k, v in self._meta_parameters.items()
        }
        gradients = None
        num_inner_steps = self._num_inner_steps
        
        for step in range(num_inner_steps):
            outputs = self._forward(images, parameters)
            frame_id_logits, player_id_logits, bb_left_logits, bb_top_logits, bb_width_logits, bb_height_logits = outputs

            # print("Logits:")
            # print(f"frame id {frame_id_logits.shape}")
            # print(f"player id {player_id_logits.shape}")
            # print(f"bb left {bb_left_logits.shape}")
            # print(f"bb top {bb_top_logits.shape}")
            # print(f"bb width {bb_width_logits.shape}")
            # print(f"bb height {bb_height_logits.shape}")
        
            # print("labels:", labels.shape)
            labels = labels[:, 0, :, :]


            frame_id_labels = labels[..., 0]
            player_id_labels = labels[..., 1]
            bb_left_labels = labels[..., 2]
            bb_top_labels = labels[..., 3]
            bb_width_labels = labels[..., 4]
            bb_height_labels = labels[..., 5]



            # print("labels:")
            # print(f"frame id {frame_id_labels.shape}")
            # print(f"player id {player_id_labels.shape}")
            # print(f"bb left {bb_left_labels.shape}")
            # print(f"bb top {bb_top_labels.shape}")
            # print(f"bb width {bb_width_labels.shape}")
            # print(f"bb height {bb_height_labels.shape}")


            frame_id_loss = F.cross_entropy(frame_id_logits.squeeze(0), frame_id_labels.squeeze(0))
            player_id_loss = F.cross_entropy(player_id_logits.squeeze(0), player_id_labels.squeeze(0))
            bb_left_loss = F.cross_entropy(bb_left_logits.squeeze(0), bb_left_labels.squeeze(0))
            bb_top_loss = F.cross_entropy(bb_top_logits.squeeze(0), bb_top_labels.squeeze(0))
            bb_width_loss = F.cross_entropy(bb_width_logits.squeeze(0), bb_width_labels.squeeze(0))
            bb_height_loss = F.cross_entropy(bb_height_logits.squeeze(0), bb_height_labels.squeeze(0))

            loss = frame_id_loss + player_id_loss + bb_left_loss + bb_top_loss + bb_width_loss + bb_height_loss

            # Calculate accuracy for each component
            accuracy_dict = {
            "frame_id": util.calculate_accuracy(frame_id_logits, frame_id_labels),
            "player_id": util.calculate_accuracy(player_id_logits, player_id_labels),
            "bb_left": util.calculate_accuracy(bb_left_logits, bb_left_labels),
            "bb_top": util.calculate_accuracy(bb_top_logits, bb_top_labels),
            "bb_width": util.calculate_accuracy(bb_width_logits, bb_width_labels),
            "bb_height": util.calculate_accuracy(bb_height_logits, bb_height_labels),
        }
            accuracies.append(accuracy_dict)
            print(f"Accuracies : {accuracy_dict}")

            # Calculate gradients using autograd.grad
            if train: 
                gradients = torch.autograd.grad(loss, parameters.values(), create_graph=True)
            else:
                gradients = torch.autograd.grad(loss, parameters.values(), create_graph=False)

            # Update parameters using gradient descent with individual inner learning rates
            for idx, key in enumerate(parameters.keys()):
                parameters[key] = parameters[key] - self._inner_lrs[key] * gradients[idx] 

        updated_outputs = self._forward(images, parameters)
        frame_id_logits2, player_id_logits2, bb_left_logits2, bb_top_logits2, bb_width_logits2, bb_height_logits2 = updated_outputs

        # print("Logits2:")
        # print(f"frame id {frame_id_logits2.shape}")
        # print(f"player id {player_id_logits2.shape}")
        # print(f"bb left {bb_left_logits2.shape}")
        # print(f"bb top {bb_top_logits2.shape}")
        # print(f"bb width {bb_width_logits2.shape}")
        # print(f"bb height {bb_height_logits2.shape}")
        
        accuracy_dict2 = {
            "frame_id": util.calculate_accuracy(frame_id_logits, frame_id_labels),
            "player_id": util.calculate_accuracy(player_id_logits, player_id_labels),
            "bb_left": util.calculate_accuracy(bb_left_logits, bb_left_labels),
            "bb_top": util.calculate_accuracy(bb_top_logits, bb_top_labels),
            "bb_width": util.calculate_accuracy(bb_width_logits, bb_width_labels),
            "bb_height": util.calculate_accuracy(bb_height_logits, bb_height_labels),
        }
        accuracies.append(accuracy_dict2)
        print(f"Accuracies 2: {accuracy_dict2}")

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

        for task_idx in range(images.size(0)):  # Iterate over tasks in the batch
            print("task num:", task_idx)
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
        for i_step, (images, labels, sports_order) in enumerate(dataloader_meta_train, start=self._start_train_step):
            print(f"Iteration: {i_step}")
            self._optimizer.zero_grad()

            for i_batch in range(images.size(0)):
                images_b = images[i_batch]  # Select one batch of images
                labels_b = labels[i_batch]  # Select one batch of labels
                outer_loss, accuracies_support, accuracy_query = self._outer_step(images_b, labels_b, train=True)
                outer_loss.backward()
            # outer_loss, accuracies_support, accuracy_query = (
            #     self._outer_step(images, labels, train=True)
            # )
            # outer_loss.backward()
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


    def test(self, dataloader_test):
        """Evaluate the MAML on test tasks using MOT metrics.

        Args:
            dataloader_test (DataLoader): Loader for test tasks.
        """
        mota_scores = []
        motp_scores = []

        for task_batch in dataloader_test:
            mota, motp = compute_mot_metrics(task_batch)
            
            mota_scores.append(mota)
            motp_scores.append(motp)

        mean_mota = np.mean(mota_scores)
        std_mota = np.std(mota_scores)
        mean_motp = np.mean(motp_scores)
        std_motp = np.std(motp_scores)

        mota_confidence_interval = 1.96 * std_mota / np.sqrt(len(mota_scores))
        motp_confidence_interval = 1.96 * std_motp / np.sqrt(len(motp_scores))

        print(
            f'MOTA over {len(mota_scores)} test tasks: '
            f'mean {mean_mota:.3f}, '
            f'95% confidence interval {mota_confidence_interval:.3f}'
        )
        
        print(
            f'MOTP over {len(motp_scores)} test tasks: '
            f'mean {mean_motp:.3f}, '
            f'95% confidence interval {motp_confidence_interval:.3f}'
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
        log_dir = f'./logs/model_{args.model}/way_{args.num_way}.support_{args.num_support}.query_{args.num_query}.inner_steps_{args.num_inner_steps}.inner_lr_{args.inner_lr}.learn_inner_lrs_{args.learn_inner_lrs}.outer_lr_{args.outer_lr}.batch_size_{args.meta_batch_size}'  # pylint: disable=line-too-long
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

<<<<<<< HEAD
    if args.checkpoint_step > -1:
        maml.load(args.checkpoint_step)
    else:
        print('Checkpoint loading skipped.')
=======
    for i in range(2):
        images, labels, random_order = next(meta_train_loader)
>>>>>>> e9ce0512a6453d6fcdf770daa8b12a5d2703b4c9



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
    parser.add_argument('--num_support', type=int, default=2, help='number of support examples per class in a task')
    parser.add_argument('--num_query', type=int, default=1, help='number of query examples per class in a task')
    parser.add_argument('--meta_batch_size', type=int, default=2, help='number of tasks per outer-loop update')
    parser.add_argument('--num_train_iterations', type=int, default=15000, help='number of outer-loop updates to train for')
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