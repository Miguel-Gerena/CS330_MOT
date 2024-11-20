# Enhancing Player Tracking Across Sports Domains Through Meta-Learning with Video Input

## Stanford University

**Authors**: Akayla Hackson, Lavinia Pedrollo, Miguel Gerena

**Contact**: [akayla@stanford.edu](mailto:akayla@stanford.edu), [laviniap@stanford.edu](mailto:laviniap@stanford.edu), [miguelg2@stanford.edu](mailto:miguelg2@stanford.edu)

---

### Abstract
This project integrates Model-Agnostic Meta-Learning (MAML) with the OSNet model within the Deep-ExpansionIoU (Deep-EIoU) tracking framework to enhance player tracking in sports. By leveraging MAML's quick adaptation capabilities and OSNet's robust feature extraction, we aim to improve tracking precision and efficiency. Our modified OSNet-MAML model achieves higher performance metrics than the state-of-the-art Deep-EIoU model across various tracking indices.

---

### Introduction
Traditional player tracking systems often suffer from inefficiencies, particularly in sports where player movement is complex and unpredictable. Our project enhances the accuracy of player tracking in sports through the innovative use of meta-learning, specifically Model-Agnostic Meta-Learning (MAML), integrated with the OSNet model for feature extraction.

---

### Model Development
- **Deep-EIoU Model**: Selected for its superior performance in sports environments, handling complex, nonlinear motion tracking without traditional methods like the Kalman filter.
- **Integration of OSNet with MAML**: Focuses on the re-identification part of the tracking algorithm, enhancing adaptability to new tasks with limited data.

---

### Methodology
- **Meta-Learning Framework**: MAML is employed to adapt the OSNet model quickly to new sports scenarios, optimizing it for scenarios where data is limited.
- **Performance Enhancement**: The OSNet-MAML model shows improvements in multiple tracking metrics (e.g., HOTA, MOTA, IDF1, DetA, AssA) over the base Deep-EIoU model.

---

### Experiments and Results
Our experiments demonstrate that the OSNet-MAML configuration outperforms traditional setups in the rigorous settings of the SportMOT dataset, particularly excelling in sports with high motion dynamics and unpredictability.

---

### Conclusion and Future Work
Our integration of MAML with OSNet not only improves tracking accuracy but also sets a new benchmark in sports analytics by adapting effectively across varied sports domains. Future work will explore expanding this model’s adaptability and incorporating additional models like YOLOX into the MAML framework for a comprehensive improvement in tracking performance.

---


# Deep-EIoU Workflow Guide

## Deep-EIoU Setup

1. **MAML Demo Execution**
   - Navigate to `Deep-EIoU/Deep-EIoU`.
   - Edit `tools/maml_demo.py`:
     - Ensure `args.path` is the path to your dataset.
     - Set `args.experiment_name` to `sportsmot-{dataset_type}`.
     - Add `args.fine_tune` if using MAML for fine-tuning.
   - Run the script to create predictions:
     ```
     python tools/maml_demo.py --path C:\Users\akayl\Desktop\CS330_MOT\dataset\train --experiment-name sportsmot-train --fine_tune
     ```
   - Results are in `YOLOX_Outputs/{experiment_name}`. Check out the video!

2. **SCT Folder Creation**
   - Stay in `Deep-EIoU/Deep-EIoU`.
   - Edit `ak_make_SCT_folder.py`:
     - Verify the paths are correct for your system.
     - Ensure `dataset_type` matches `{dataset_type}` in `args.experiment_name` from `maml_demo.py`.
   - Run the command to create the SCT folder in `CS330_MOT/Deep-EIoU`:
     ```
     python ak_make_SCT_folder.py
     ```

3. **Sport Interpolation**
   - Still in `Deep-EIoU/Deep-EIoU`.
   - Edit `tools/sport_interpolation.py`:
     - Check `args.dataset_type` is correct.
   - Execute to create an interpolation folder in `CS330_MOT/Deep-EIoU`, formatting the SCT folder for TrackEval:
     ```
     python tools/sport_interpolation.py
     ```

## TrackEval Setup (Manual)

- Navigate to `CS330_MOT/TrackEval/data`.
  - The `ref` folder contains all reference files.
    - Ensure `sportsmot-test`, `sportsmot-train`, `sportsmot-val` have the correct videos.
    - Each should contain a `gt` folder and `seqinfo.ini`.
  - In the `seqmaps` folder, verify `.txt` files (`sportsmot-test.txt`, `sportsmot-train.txt`, `sportsmot-val.txt`).
  - Contents in the `res` folder can be deleted.

## TrackEval Execution

1. **Interpolation to Res**
   - Go to `CS330_MOT`.
   - Edit `ak_util/interpolation_to_res.py`:
     - Set correct paths and `dataset_type`.
   - Run to move predictions to `TrackEval/data/res`:
     ```
     python ak_util/interpolation_to_res.py
     ```

2. **TrackEval Evaluation**
   - Navigate to `TrackEval`.
   - Update the `evaluate.sh` file:
     - Change `SPLIT_TO_EVAL` and `SEQMAP_FILE` to the correct `dataset_type`.
   - Run to get tracking results:
     ```
     python ./scripts/run_mot_challenge.py --BENCHMARK sportsmot --SPLIT_TO_EVAL test --METRICS HOTA CLEAR Identity VACE --USE_PARALLEL False --PRINT_CONFIG True --GT_FOLDER ./data/ref --TRACKERS_FOLDER ./data/res --OUTPUT_FOLDER ./output/ --SEQMAP_FILE ./data/ref/seqmaps/sportsmot-test.txt
     ```
   - Results are in `TrackEval/output`.

**All done!**
