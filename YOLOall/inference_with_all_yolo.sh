#!/bin/bash

# Function to execute YOLO model inference
run_yolo() {
    echo "Running YOLOv$1 inference..."
    wget $2
    $3 detect $4 $5 $6 $7 $8
    echo "YOLOv$1 inference complete."
    echo ""
}

# Prerequisites
# echo "Setting up prerequisites..."
# sudo apt-get install git  # Assuming a Debian-based system; adjust for your OS
# sudo apt-get install make
# sudo apt-get install wget
# echo "Prerequisites installed."

# Clone the YOLOv2 repository
echo "Running YOLOv2..."
git clone https://github.com/AlexeyAB/darknet.git
cd darknet
make
cd ..
wget https://pjreddie.com/media/files/yolov2.weights
./darknet detect cfg/yolov2.cfg yolov2.weights "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/000002.jpg"
echo "YOLOv2 complete."
echo ""

# Run YOLOv3
run_yolo 3 https://pjreddie.com/media/files/yolov3.weights ./darknet detect cfg/yolov3.cfg yolov3.weights "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/000002.jpg"

# Run YOLOv3-tiny
run_yolo 3 https://pjreddie.com/media/files/yolov3-tiny.weights ./darknet detect cfg/yolov3-tiny.cfg yolov3-tiny.weights "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/000002.jpg"

# Run YOLOv4
run_yolo 4 https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights ./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/000002.jpg"
cd ..

# Run YOLOv5
echo "Running YOLOv5..."
pip install ultralytics
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
python detect.py --weights yolov5s.pt --source "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/000002.jpg" 
python detect.py --weights yolov5l.pt --source "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/000002.jpg" 
python detect.py --weights yolov5n.pt --source "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/000002.jpg" 
python detect.py --weights yolov5m.pt --source "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/000002.jpg" 
python detect.py --weights yolov5x.pt --source "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/000002.jpg" 
cd ..

# Run YOLOv6
echo "Running YOLOv6..."
git clone https://github.com/meituan/YOLOv6
cd YOLOv6
pip install -r requirements.txt
wget https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6lite_s.pt
python tools/infer.py --weights yolov6s.pt --source "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/000002.jpg" 
python tools/infer.py --weights yolov6m.pt --source "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/000002.jpg" 
python tools/infer.py --weights yolov6l.pt --source "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/000002.jpg" 
cd ..

# Run YOLOv7
echo "Running YOLOv7..."
git clone https://github.com/WongKinYiu/yolov7
cd yolov7
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6.pt
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-seg.pt
python detect.py --source "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/000002.jpg" --weights yolov7-tiny.pt --view-img
python detect.py --source "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/000002.jpg" --weights yolov7.pt --view-img
python detect.py --source "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/000002.jpg" --weights yolov7x.pt --view-img
python detect.py --source "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/000002.jpg" --weights yolov7-w6.pt --view-img
python detect.py --source "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/000002.jpg" --weights yolov7-e6.pt --view-img
python detect.py --source "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/000002.jpg" --weights yolov7-d6.pt --view-img
python detect.py --source "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/000002.jpg" --weights yolov7-e6e.pt --view-img
python detect.py --source "/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/000002.jpg" --weights yolov7-seg.pt --view-img
echo "YOLOv7 complete."
echo ""

# Run YOLOv8
echo "Running YOLOv8..."
cd ..
pip install ultralytics
# Instructions to Run YOLOv8
pip install ultralytics

# Download YOLOv8 weights
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt

# Run YOLOv8 inference on the specified image for various weights
yolo predict model=yolov8n.pt source="/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/000002.jpg" 
yolo predict model=yolov8s.pt source="/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/000002.jpg"
yolo predict model=yolov8m.pt source="/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/000002.jpg"
yolo predict model=yolov8l.pt source="/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/000002.jpg"
yolo predict model=yolov8x.pt source="/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/dataset/train/v_-6Os86HzwCs_c009/img1/000002.jpg"
