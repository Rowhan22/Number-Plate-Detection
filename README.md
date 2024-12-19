# Number-Plate-Detection

# Project Overview
This project aims to detect vehicle license plates and classify the type of vehicle based on the color of its license plate. It uses computer vision techniques to identify license plates in real-time video streams, extract the region of interest (ROI), and analyze the color information to determine the vehicle category, such as "Electric vehicle","Private ownership" etc.The project integrates OpenCV for image processing and a pre-trained Haar Cascade XML classifier for license plate detection.

# Getting Started
1. Clone the repository or download the files.
2. Install dependencies:
   pip install opencv-python numpy  
3. Connect a webcam and ensure number_plate_detection.xml is in the project directory.

# Usage
1. Run the script:
   python test.py  
2. Point the webcam at a vehicle license plate.
   The detected license plate color and vehicle type will be displayed.
3. Press q to exit.

# Prerequisites
Python 3.x
Libraries: opencv-python, numpy
Webcam for real-time detection
Minimum 4GB RAM for smooth processing
