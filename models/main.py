import argparse

import os
import importlib.util
import random
import torch
import numpy as np
import multiprocessing
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sys

sys.path.append('./data/')
from blackbox import Black
from load_data import DataGenerator


# Check if submission module is present.  If it is not, then main() will not be executed.

def meta_train_step(images, labels, model, optim, eval=False):

    predictions = model(images, labels)
    loss = model.loss_function(predictions, labels)
    if not eval:
        optim.zero_grad()
        loss.backward()
        optim.step()
    return predictions.detach(), loss.detach()


def main(config):
    print(config)
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    
    if config.device == "gpu" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif config.device == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device: ", device)

    torch.manual_seed(config.random_seed)

    writer = SummaryWriter(f"runs/{config.num_way}_{config.num_support}_{config.random_seed}_{config.hidden_dim}_{config.outer_lr}")

    # Create Data Generator
    # This will sample meta-training and meta-testing tasks

    meta_train_iterable = DataGenerator(
        config.num_videos,
        config.num_support + config.num_query,
        batch_type="train",
        cache=config.cache,  # false by default
        generate_new_tasks=True,
    )
    meta_train_loader = iter(
        torch.utils.data.DataLoader(
            meta_train_iterable,
            batch_size=config.meta_batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
        )
    )

    # meta_test_iterable = DataGenerator(
    #     config.num_way,
    #     config.num_shot + 1,
    #     batch_type="test",
    #     cache=config.image_caching,
    # )
    # meta_test_loader = iter(
    #     torch.utils.data.DataLoader(
    #         meta_test_iterable,
    #         batch_size=config.meta_batch_size,
    #         num_workers=config.num_workers,
    #         pin_memory=True,
    #     )
    # )

    # Create model
    model = Black(config.num_way, config.num_support, config.num_query, config.hidden_dim)

    # if(config.compile == True):
    #     try:
    #         model = torch.compile(model, backend=config.backend)
    #         print(f"MANN model compiled")
    #     except Exception as err:
    #         print(f"Model compile not supported: {err}")

    model.to(device)

    # Create optimizer
    optim = torch.optim.Adam(model.parameters(), lr=config.outer_lr)
    import time

    times = []
    
    for step in tqdm(range(config.num_train_iterations)):
        ## Sample Batch
        ## Sample some meta-training tasks
        t0 = time.time()
        images, labels, random_order = next(meta_train_loader)
        images, labels = images.to(device), labels.to(device)
        t1 = time.time()

        ## Train
        prediction, loss = meta_train_step(images, labels, model, optim)
        t2 = time.time()
        writer.add_scalar("Loss/train", loss, step)
        times.append([t1 - t0, t2 - t1])

        ## Evaluate
        ## Get meta-testing tasks
        # if (step + 1) % config.eval_freq == 0:
        #     if config.debug == True:
        #         print("*" * 5 + "Iter " + str(step + 1) + "*" * 5)
        #     i, l = next(meta_test_loader)
        #     i, l = i.to(device), l.to(device)
        #     pred, tls = meta_train_step(i, l, model, optim, eval=True)
        #     if config.debug == True:
        #         print("Train Loss:", ls.cpu().numpy(), "Test Loss:", tls.cpu().numpy())
        #     writer.add_scalar("Loss/test", tls, step)
        #     pred = torch.reshape(
        #         pred, [-1, config.num_shot + 1, config.num_way, config.num_way]
        #     )

        #     with open(f'submission/mann_results_{config.num_shot}_{config.num_way}_{config.hidden_dim}_{config.outer_lr}.npy', 'wb') as f:
        #         np.save(f, l.cpu().numpy())
        #         np.save(f, pred.cpu().numpy())

        #     pred = torch.argmax(pred[:, -1, :, :], axis=2)
        #     l = torch.argmax(l[:, -1, :, :], axis=2)
        #     acc = pred.eq(l).sum().item() / (config.meta_batch_size * config.num_way)
        #     if config.debug == True:
        #         print("Test Accuracy", acc)
        #     writer.add_scalar("Accuracy/test", acc, step)

        #     times = np.array(times)
        #     if config.debug == True:
        #         print(f"Sample time {times[:, 0].mean()} Train time {times[:, 1].mean()}")
        #     times = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default=None,
                        help='directory to save to or load from')
    parser.add_argument('--model', type=str, default='maml',
                        help='model to run')
    parser.add_argument('--frame_selection_strategy', type=str, default="consecutive_frame_selection",
                        help='The function in load_by_sport file that we want to use for splitting the support/query sets. EX: consecutive vs random')
    parser.add_argument('--num_videos', type=int, default=1,
                        help='number of videos to include in the support set')
    parser.add_argument('--num_query_videos', type=int, default=2,
                        help='number of videos to include in the query set')
    parser.add_argument('--num_way', type=int, default=23,
                        help='number of classes in a task')
    parser.add_argument('--num_support', type=int, default=2,
                        help='number of support examples per class in a task')
    parser.add_argument('--num_query', type=int, default=1,
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
    parser.add_argument('--meta_batch_size', type=int, default=2,
                        help='number of tasks per outer-loop update')
    parser.add_argument('--num_train_iterations', type=int, default=15000,
                        help='number of outer-loop updates to train for')
    parser.add_argument('--test', default=False, action='store_true',
                        help='train or test')
    parser.add_argument('--num_workers', type=int, default=int(multiprocessing.cpu_count()/2), 
                        help=('needed to specify dataloader'))
    parser.add_argument('--checkpoint_step', type=int, default=-1,
                        help=('checkpoint iteration to load for resuming '
                            'training, or for evaluation (-1 is ignored)'))
    parser.add_argument('--random_seed', type=int, default=0,
                        help=('set random seed'))
    parser.add_argument('--hidden_dim', type=int, default=100,
                        help=('Number of hidden dimensions for the network'))
    parser.add_argument('--cache', type = bool, default=False)
    parser.add_argument('--device', type=str, default='gpu')
    args = parser.parse_args()
    main(args)
