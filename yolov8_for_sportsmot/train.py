from ultralytics import YOLO
#
# from roboflow import Roboflow
# rf = Roboflow(api_key="Kh47OhEcvKoHpuitcBWi")
# project = rf.workspace("cs330").project("ddd-bjl64")
# dataset = project.version(1).download("yolov8")

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
yaml = r"/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/Project Repository/CS330_MOT/data/sportsmot_publish/dataset/Sportsmot_dataset_yolov8/spotsmot_yolov8Format.yaml"
# Use the model
model.train(data=yaml, epochs=3, imgsz=640, device='mps')  # train the model
model.save("modelsave")
metrics = model.val()  # evaluate model performance on the validation set
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format

#
# yolo task=detect \
# mode=train \
# model=yolov8s.pt \
# data=r"/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/Project Repository/CS330_MOT/data/sportsmot_publish/dataset/Sportsmot_dataset_yolov8/spotsmot_yolov8Format.yaml" \
# epochs=100