import torch
from PIL import Image
from pathlib import Path
from argparse import ArgumentParser
from ultralytics import YOLO
import torch
from models import YOLO  # Import your YOLO model implementation

# Create an instance of your YOLO model
model = YOLO()

# Load the trained weights
weights_path = 'path/to/your/trained_weights.pt'  # Replace with the path to your trained weights file
model.load_state_dict(torch.load(weights_path))

# Set the model to evaluation mode
model.eval()

# Optionally, move the model to a specific device (CPU/GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Now you can use this model for inference or further training

def detect(image_path, weights="yolov8n.pt", output_dir="./inference/output"):
    # Load the model
    model = weights

    # Load the image
    img = Image.open(image_path)

    # Perform inference
    results = model(img)

    # Print results
    print(results)

    # Save results
    results.save(Path(output_dir) / Path(image_path).stem)  # save to ./inference/output

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--image', type=str, default = r"/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/Project Repository/CS330_MOT/data/sportsmot_publish/dataset/test/v_iF9bKPWdZlc_c001/img1/000001.jpg", help='path to image file')
    parser.add_argument('--weights', type=str, default=r"/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/Project Repository/CS330_MOT/yolov8_for_sportsmot/best.pt", help='path to model weights')
    parser.add_argument('--output_dir', type=str, default='./inference/output', help='output directory for saving results')
    args = parser.parse_args()

    detect(args.image, args.weights, args.output_dir)

# python test.py --image r"/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/Project Repository/CS330_MOT/data/sportsmot_publish/dataset/test/v_-9kabh1K8UA_c008/img1/000001.jpg" --weights r"/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/Project Repository/CS330_MOT/yolov8_for_sportsmot/best.pt" --output_dir /output_directory