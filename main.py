import random
import argparse
import json
import sys

import torch 
import tensorboard
sys.path.append('./ak_util/')
import load_by_sport






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

    # Get support/query sets based off args
    sports_dict = load_by_sport.get_videos_file_path_dict(args.counts_json_path, args.dataset_folder_path)
    sports_dict = load_by_sport.get_videos_file_path_dict(args.counts_json_path, args.dataset_folder_path)
    support, query = load_by_sport.create_support_query_split(sports_dict, args.num_support, args.num_query, frame_selection_strategy=strategy_function)
    # print(f"\n\nJUST FINISHED MAKING SUPPORT AND QUERY SETS:\nSupport set: {support}\nQuery set: {query}")





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts_json_path", default='./train_counts.json', 
                        help="Path to the JSON file containing video categorization.")
    parser.add_argument("--dataset_folder_path", default='./dataset/train/', 
                        help="Path to the data folder.")
    parser.add_argument('--log_dir', type=str, default=None,
                        help='directory to save to or load from')
    parser.add_argument('--model', type=str, default='maml',
                        help='model to run')
    parser.add_argument('--frame_selection_strategy', type=str, default="consecutive_frame_selection",
                        help='The function in load_by_sport file that we want to use for splitting the support/query sets. EX: consecutive vs random')
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
    parser.add_argument('--batch_size', type=int, default=15,
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