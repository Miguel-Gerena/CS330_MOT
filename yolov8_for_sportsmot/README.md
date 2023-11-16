# SportsMot Dataset Conversion to YOLOv8 Format and Training on YOLOv8

This folder contains scripts to convert annotations from the SportsMot dataset into YOLO format and to train YOLOv8 using this converted data.

## File Structure

- `mot_to_yolov8.py`: Python script for converting SportsMot annotations to YOLO format.
- `train_yolov8.py`: Python script for training YOLOv8 on the converted dataset.

## Instructions

### `mot_to_yolov8.py`

#### Purpose
This script is used to convert annotations from the SportsMot dataset to the YOLO format, which is required for training YOLOv8.

#### Usage
1. Modify the paths:
    - `DATA_PATH`: Change this to the base path where your SportsMot dataset is located.
    - `OUT_PATH`: Output directory where the converted dataset will be saved.
    - `SPLITS`: Specify the dataset splits to process (`train`, `val`, `test`).
    - `HALF_VIDEO`: Set to `True` if processing half video sequences.

2. Functions:
    - `convert_annotation`: Converts SportsMot annotation to YOLO format.
    - `loop over the splits`: Processes images and annotations based on the specified splits.

### `train_yolov8.py`

#### Purpose
This script is used to train YOLOv8 on the converted SportsMot dataset.

#### Usage
1. Set the model:
    - Choose between building a new model or loading a pretrained model by uncommenting the respective lines (`model = YOLO("yolov8n.yaml")` or `model = YOLO("yolov8n.pt")`).

2. Specify the YAML configuration:
    - Update `yaml` variable with the path to your YOLOv8 configuration file.

3. Training:
    - `model.train`: Trains the model for the specified number of epochs on the dataset.
    - `model.save`: Saves the trained model.
    - `metrics = model.val()`: Evaluates model performance on the validation set.
    - `path = model.export`: Exports the trained model in ONNX format.

## Important Notes

- **Dataset Path:** Ensure to change the `DATA_PATH` variable in `mot_to_yolov8.py` to the correct path where your SportsMot dataset is located.
- **Model Configuration:** Modify the YOLOv8 configuration file path (`yaml`) in `train_yolov8.py` based on your setup.
- **Training Parameters:** Adjust the training parameters (`epochs`, `device`) according to wanted experiments (see YOLOv8 documentation for full range of changeable parameters).
