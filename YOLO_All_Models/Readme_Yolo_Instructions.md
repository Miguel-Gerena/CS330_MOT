# YOLO Models Inference Instructions

## Models explored:
YOLOv2, YOLOv3, YOLOv3-tiny, YOLOv4, YOLOv5, YOLOv6, YOLOv7, YOLOv8 as specified in their respective instructions.

## Instructions to Run YOLOv2
```bash
# Clone the YOLOv2 repository
git clone https://github.com/AlexeyAB/darknet.git
chmod 777 darknet/
# Navigate to the darknet directory
cd darknet
# Build darknet
make
# Download YOLOv2 weights
wget https://pjreddie.com/media/files/yolov2.weights
# Run YOLOv2 inference on the specified image
./darknet detect cfg/yolov2.cfg yolov2.weights "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/000002.jpg"
cd ..
```





## Instructions to Run YOLOv3
```bash
# Be sure you are still in darknet folder
cd darknet
# Download YOLOv3 weights
wget https://pjreddie.com/media/files/yolov3.weights 
# Run YOLOv3 inference on the specified image
./darknet detect cfg/yolov3.cfg yolov3.weights "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/000002.jpg"
cd ..
```

## Instructions to Run YOLOv3-tiny

```bash
cd darknet
# Download YOLOv3-tiny weights
wget https://pjreddie.com/media/files/yolov3-tiny.weights
# Run YOLOv3-tiny inference on the specified image
./darknet detect cfg/yolov3-tiny.cfg yolov3-tiny.weights "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/000002.jpg"
cd ..
```


## Instructions to Run YOLOv4
```bash
# Download YOLOv4 weights
cd darknet
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
# Run YOLOv4 inference on the specified image
./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/000002.jpg"
# Move back to the parent directory
cd ..

```
## Instructions to Run YOLOv5

```bash
# Install Ultralytics
pip install ultralytics
# Clone YOLOv5 repository
git clone https://github.com/ultralytics/yolov5
# Navigate to the yolov5 directory
cd yolov5
# Install YOLOv5 dependencies
pip install -r requirements.txt
# Run YOLOv5 inference on the specified image for various weights
python detect.py --weights yolov5s.pt --source "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/000002.jpg" 
# Repeat the above command for yolov5l, yolov5n, yolov5m, and yolov5x
cd ..

```


## Instructions to Run YOLOv6
```bash
# Clone YOLOv6 repository
git clone https://github.com/meituan/YOLOv6
# Navigate to the YOLOv6 directory
cd YOLOv6
# Install YOLOv6 dependencies
pip install -r requirements.txt
# Download YOLOv6-lite weights
wget https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6lite_s.pt
# Run YOLOv6 inference on the specified image for various weights
python tools/infer.py --weights yolov6s.pt --source "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/000002.jpg" 
# Repeat the above command for yolov6m, yolov6l, and other weights
cd ..

```

## Instructions to Run YOLOv7
```bash
# Clone YOLOv7 repository
git clone https://github.com/WongKinYiu/yolov7
# Navigate to the yolov7 directory
cd /yolov7
# Download YOLOv7 weights
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
# Run YOLOv7 inference on the specified image for various weights
python detect.py --source "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/000002.jpg"  --weights yolov7-tiny.pt --view-img
# Repeat the above command for yolov7.pt, yolov7x.pt, yolov7-w6.pt, yolov7-e6.pt, yolov7-d6.pt, yolov7-e6e.pt, yolov7-seg.pt
cd ..
```

## Instructions to Run YOLOv8
```bash
# Install Ultralytics
pip install ultralytics
mkdir yolov8
cd yolov8
# Download YOLOv8 weights
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt
# Run YOLOv8 inference on the specified image for various weights
yolo predict model=yolov8n.pt source="/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/000002.jpg"

``` 






<!-- cd YOLO_All_Models 
git clone https://github.com/pjreddie/darknet
git clone https://github.com/AlexeyAB/darknet.git
cd darknet
make
YOLOv3 is extremely fast and accurate. In mAP measured at .5 IOU YOLOv3 is on par with Focal Loss but about 4x faster. Moreover, you can easily tradeoff between speed and accuracy simply by changing the size of the model, no retraining required!
wget https://pjreddie.com/media/files/yolov3.weights
./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg
Darknet prints out the objects it detected, its confidence, and how long it took to find them. We didn't compile Darknet with OpenCV so it can't display the detections directly. Instead, it saves them in predictions.png. You can open it to see the detected objects. Since we are using Darknet on the CPU it takes around 6-12 seconds per image. If we use the GPU version it would be much faster.

I've included some example images to try in case you need inspiration. Try data/eagle.jpg, data/dog.jpg, data/person.jpg, or data/horses.jpg!

The detect command is shorthand for a more general version of the command. It is equivalent to the command:

./darknet detector test cfg/coco.data cfg/yolov3.cfg yolov3.weights data/dog.jpg
You don't need to know this if all you want to do is run detection on one image but it's useful to know if you want to do other things like run on a webcam (which you will see later on).

Multiple Images

Instead of supplying an image on the command line, you can leave it blank to try multiple images in a row. Instead you will see a prompt when the config and weights are done loading:

Changing The Detection Threshold

By default, YOLO only displays objects detected with a confidence of .25 or higher. You can change this by passing the -thresh <val> flag to the yolo command. For example, to display all detection you can set the threshold to 0:

./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg -thresh 0


Tiny YOLOv3

We have a very small model as well for constrained environments, yolov3-tiny. To use this model, first download the weights:

wget https://pjreddie.com/media/files/yolov3-tiny.weights
Then run the detector with the tiny config file and weights:

./darknet detect cfg/yolov3-tiny.cfg yolov3-tiny.weights data/dog.jpg


/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/predictions_yolov5_v_-6Os86HzwCs_c009_img1_000002_jpg.jpg








Path to your image: "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/predictions_yolov5_v_-6Os86HzwCs_c009_img1_000002_jpg.jpg" 

git clone https://github.com/AlexeyAB/darknet.git
cd darknet
make
cd ..

INSTRUCTIONS to run YOLOv2
wget https://pjreddie.com/media/files/yolov2.weights
./darknet detect cfg/yolov2.cfg yolov2.weights "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/predictions_yolov5_v_-6Os86HzwCs_c009_img1_000002_jpg.jpg"


INSTRUCTIONS to run YOLOv3
wget https://pjreddie.com/media/files/yolov3.weights 
./darknet detect cfg/yolov3.cfg yolov3.weights "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/predictions_yolov5_v_-6Os86HzwCs_c009_img1_000002_jpg.jpg"

INSTRUCTIONS to run YOLOv3-tiny
wget https://pjreddie.com/media/files/yolov3-tiny.weights
./darknet detect cfg/yolov3-tiny.cfg yolov3-tiny.weights "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/predictions_yolov5_v_-6Os86HzwCs_c009_img1_000002_jpg.jpg"

INSTRUCTIONS to run YOLOv4
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights

./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/predictions_yolov5_v_-6Os86HzwCs_c009_img1_000002_jpg.jpg"

cd ..

INSTRUCTIONS to run YOLOv5 
pip install ultralytics
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt
python detect.py --weights yolov5s.pt --source "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/predictions_yolov5_v_-6Os86HzwCs_c009_img1_000002_jpg.jpg" 
python detect.py --weights yolov5l.pt --source "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/predictions_yolov5_v_-6Os86HzwCs_c009_img1_000002_jpg.jpg" 

python detect.py --weights yolov5n.pt --source "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/predictions_yolov5_v_-6Os86HzwCs_c009_img1_000002_jpg.jpg" 

python detect.py --weights yolov5m.pt --source "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/predictions_yolov5_v_-6Os86HzwCs_c009_img1_000002_jpg.jpg" 
python detect.py --weights yolov5x.pt --source "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/predictions_yolov5_v_-6Os86HzwCs_c009_img1_000002_jpg.jpg" 


cd ..
INSTRUCTIONS to run YOLOv6
git clone https://github.com/meituan/YOLOv6
cd YOLOv6
pip install -r requirements.txt

wget https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6lite_s.pt

wget https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6lite_m.pt
wget https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6lite_l.pt

python tools/infer.py --weights yolov6s.pt --source "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/predictions_yolov5_v_-6Os86HzwCs_c009_img1_000002_jpg.jpg" 
python tools/infer.py --weights yolov6m.pt --source "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/predictions_yolov5_v_-6Os86HzwCs_c009_img1_000002_jpg.jpg" 
python tools/infer.py --weights yolov6l.pt --source "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/predictions_yolov5_v_-6Os86HzwCs_c009_img1_000002_jpg.jpg" 


INSTRUCTIONS to run YOLOv7
cd ..
git clone https://github.com/WongKinYiu/yolov7
cd /yolov7

wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt

wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6.pt
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-seg.pt


python detect.py --source "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/predictions_yolov5_v_-6Os86HzwCs_c009_img1_000002_jpg.jpg" --weights yolov7-tiny.pt --view-img
python detect.py --source "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/predictions_yolov5_v_-6Os86HzwCs_c009_img1_000002_jpg.jpg" --weights yolov7.pt --view-img
python detect.py --source "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/predictions_yolov5_v_-6Os86HzwCs_c009_img1_000002_jpg.jpg" --weights yolov7x.pt --view-img
python detect.py --source "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/predictions_yolov5_v_-6Os86HzwCs_c009_img1_000002_jpg.jpg" --weights yolov7-w6.pt --view-img
python detect.py --source "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/predictions_yolov5_v_-6Os86HzwCs_c009_img1_000002_jpg.jpg" --weights yolov7-e6.pt --view-img
python detect.py --source "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/predictions_yolov5_v_-6Os86HzwCs_c009_img1_000002_jpg.jpg" --weights yolov7-d6.pt --view-img
python detect.py --source "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/predictions_yolov5_v_-6Os86HzwCs_c009_img1_000002_jpg.jpg" --weights yolov7-e6e.pt --view-img
python detect.py --source "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/predictions_yolov5_v_-6Os86HzwCs_c009_img1_000002_jpg.jpg" --weights yolov7-seg.pt --view-img




cd ..
pip install ultralytics

wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt


yolo predict model=yolov8n.pt source="/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/predictions_yolov5_v_-6Os86HzwCs_c009_img1_000002_jpg.jpg" 
yolo predict model=yolov8s.pt source="/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/predictions_yolov5_v_-6Os86HzwCs_c009_img1_000002_jpg.jpg
yolo predict model=yolov8m.pt source="/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/predictions_yolov5_v_-6Os86HzwCs_c009_img1_000002_jpg.jpg
yolo predict model=yolov8l.pt source="/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/predictions_yolov5_v_-6Os86HzwCs_c009_img1_000002_jpg.jpg
yolo predict model=yolov8x.pt source="/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/predictions_yolov5_v_-6Os86HzwCs_c009_img1_000002_jpg.jpg -->