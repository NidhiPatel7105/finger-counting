# Face & Hand Detection with Finger Counting

## Overview

This project utilizes **OpenCV** and **MediaPipe** to detect faces and hands in real-time using a webcam. It also counts the number of fingers raised on a detected hand.

## Features

- **Real-time face detection** with a bounding box
- **Real-time hand detection** with a bounding box
- **Finger counting algorithm** to detect raised fingers
- **Live video feed processing**
- **Easy-to-use and efficient implementation**

## Technologies Used

- **Python**
- **OpenCV**
- **MediaPipe**

## Installation

Ensure you have Python installed, then install the required dependencies:

```sh
pip install opencv-python mediapipe
```

## Usage

Run the script using:

```sh
python script.py
```

### Controls

- Press **'q'** to exit the program.

## How It Works

1. Captures video from the webcam.
2. Converts frames to RGB for processing.
3. Detects faces using **MediaPipe FaceDetection** and draws a bounding box.
4. Detects hands using **MediaPipe Hands** and draws a bounding box.
5. Counts the number of raised fingers based on landmark positions.
6. Displays the processed frame with bounding boxes and finger count.

