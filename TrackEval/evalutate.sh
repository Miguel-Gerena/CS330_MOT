# First, change cwd to path/to/trackeval
# cd .
# donnot forget to activate the environment if you want to use anaconda
# source activate trackeval
# call the main evalualtion process.
# python ./scripts/run_mot_challenge.py --BENCHMARK sportsmot --SPLIT_TO_EVAL test --METRICS HOTA CLEAR Identity VACE --USE_PARALLEL False --PRINT_CONFIG True --GT_FOLDER ./data/ref/sportsmot-test --TRACKERS_FOLDER ./data/res/sportsmot-test/tracker_to_eval/data --OUTPUT_FOLDER ./output/

 python ./scripts/run_mot_challenge.py --BENCHMARK sportsmot --SPLIT_TO_EVAL train --METRICS HOTA CLEAR Identity VACE --USE_PARALLEL False --PRINT_CONFIG True --GT_FOLDER ./data/ref --TRACKERS_FOLDER ./data/res --OUTPUT_FOLDER ./output/ --SEQMAP_FILE ./data/ref/seqmaps/sportsmot-train.txt