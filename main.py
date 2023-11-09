import random
import argparse
import json
import sys

import torch 
import tensorboard
sys.path.append('./')
import maml
sys.path.append('./ak_util/')
import load_by_sport



TRAIN_COUNTS_JSON_PATH = './train_counts.json'
VAL_COUNTS_JSON_PATH = './val_counts.json'
TEST_COUNTS_JSON_PATH = './test_counts.json'

TRAIN_DATASET_PATH = './dataset/train/'
VAL_DATASET_PATH = './dataset/val/'
TEST_DATASET_PATH = './dataset/test/'

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


    # log_dir = args.log_dir
    # if log_dir is None:
    #     log_dir = f'./logs/model_{args.model}/way_{args.num_way}.support_{args.num_support}.query_{args.num_query}.inner_steps_{args.num_inner_steps}.inner_lr_{args.inner_lr}.learn_inner_lrs_{args.learn_inner_lrs}.outer_lr_{args.outer_lr}.batch_size_{args.batch_size}'  # pylint: disable=line-too-long
    # print(f'log_dir: {log_dir}')
    # writer = tensorboard.SummaryWriter(log_dir=log_dir)

    # if args.model == 'maml':
    #     maml = MAML(
    #     args.num_way,
    #     args.num_inner_steps,
    #     args.inner_lr,
    #     args.learn_inner_lrs,
    #     args.outer_lr,
    #     log_dir,
    #     DEVICE
    # )
    # else:
    #     print(f"ERROR: Model '{args.model}' is not implemented yet")
    #     exit()

    # if args.checkpoint_step > -1:
    #     maml.load(args.checkpoint_step)
    # else:
    #     print('Checkpoint loading skipped.')


    if not args.test:
        # Load the train dictionary
        train_sports_dict = load_by_sport.get_videos_file_path_dict(TRAIN_COUNTS_JSON_PATH, TRAIN_DATASET_PATH)

        train_tasks = load_by_sport.get_batches(train_sports_dict, args.batch_size, args.num_support_videos, args.num_query_videos, args.num_support, args.num_query, strategy_function)
        print(f"Tasks: {len(train_tasks)}")

        # val_sports_dict = load_by_sport.get_videos_file_path_dict(VAL_COUNTS_JSON_PATH, VAL_DATASET_PATH)
        # val_tasks = load_by_sport.get_batches(val_sports_dict, args.batch_size * 4, args.num_support_videos, args.num_query_videos, args.num_support, args.num_query, strategy_function)
        # print(f"\n\n Tasks: {len(val_tasks)}")

    # if args.model == 'maml':
    #     maml.train(
    #         train_tasks,
    #         val_tasks,
    #         writer
    #     )

    else:
        pass
    

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
    parser.add_argument('--num_inner_steps', type=int, default=1,
                        help='number of inner-loop updates')
    parser.add_argument('--inner_lr', type=float, default=0.4,
                        help='inner-loop learning rate initialization')
    parser.add_argument('--learn_inner_lrs', default=False, action='store_true',
                        help='whether to optimize inner-loop learning rates')
    parser.add_argument('--outer_lr', type=float, default=0.001,
                        help='outer-loop learning rate')
    parser.add_argument('--batch_size', type=int, default=10,
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