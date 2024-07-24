# Automatic License Plate Detection Project ANPR
![License Plate Recognition](examples/ANPR.webp)

## Overview

This project focuses on developing a comprehensive parking management system that utilizes deep learning and machine learning techniques to detect and read vehicle license plates. The system is capable of processing real-time video streams from parking lot cameras using RTSP or from the computer's own camera, recording entry and exit times of vehicles into a database.

The main goal of this project is to automate the process of monitoring vehicle movements in a parking lot, providing a seamless and efficient way to manage parking spaces and track vehicle activities.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [System Architecture](#system-architecture)
- [Live Streaming and Detection](#live-streaming-and-detection)
- [Database Integration](#database-integration)
- [Example Outputs](#example-outputs)
- [How It Works](#how-it-works)
- [Future Work](#future-work)
- [License](#license)
- [Contact](#contact)

## Features

- **Real-time License Plate Detection:** Uses live video feeds to detect and recognize license plates in real-time.
- **Database Logging:** Automatically logs vehicle entry and exit times into an SQL database.
- **RTSP and Local Camera Support:** Can connect to RTSP streams from IP cameras or use the local camera for live detection.
- **Accurate Recognition:** Leverages advanced machine learning models for precise recognition.
- **Scalable Solution:** Easily integrates with existing parking management systems.

## Technologies Used

- **Deep Learning:** Implements Convolutional Neural Networks (CNNs) for detecting and recognizing license plates.
- **YOLOv8:** Utilizes the YOLOv8 architecture for fast and accurate object detection.
- **OpenCV:** Handles image processing tasks, such as capturing video frames and manipulating images.
- **Python:** The main programming language for the project.
- **SQL:** Used to create and manage the database for recording vehicle entry and exit times.
- **PyTorch:** The deep learning framework for building and training models.
- **RTSP Protocol:** Enables streaming from IP cameras for real-time detection.

## Installation

To set up the project on your local machine, follow these steps:

### Clone the Repository

```bash
git clone https://github.com/yourusername/parking-management-system.git
cd parking-management-system
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Ensure that you have Python 3.7 or higher installed on your system. Additionally, you'll need access to an IP camera supporting RTSP streaming or a computer with a webcam.

### Database Setup

You'll need to set up an SQL database to store the vehicle entry and exit times. Here's a simple example using SQLite:

```sql
CREATE TABLE parking_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    license_plate TEXT NOT NULL,
    entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    exit_time TIMESTAMP
);
```

### Update Database Configuration

Edit the `config.py` file to specify your database connection details.

```python
DATABASE_URI = 'sqlite:///parking.db'
```

For other database systems like MySQL or PostgreSQL, update the URI accordingly.

## System Architecture

The system consists of several components working together to achieve efficient license plate detection and management:

1. **Video Stream Input:** Captures video frames from RTSP cameras or a local webcam.
2. **License Plate Detection:** Processes frames to detect license plates using the YOLOv8 model.
3. **License Plate Recognition:** Reads the detected license plates using Optical Character Recognition (OCR).
4. **Database Logging:** Stores the entry and exit times of vehicles in an SQL database.
5. **User Interface:** Displays detected license plates and database records (optional).

### Architecture Diagram

```plaintext
[Camera/RTSP] --> [Video Stream] --> [License Plate Detection] --> [OCR] --> [Database Logging]
     |                                                                        |
     +--------------------------------[User Interface]------------------------+
```

## Live Streaming and Detection

The system is capable of connecting to live video streams via RTSP or using the computer's own camera for real-time license plate detection. This feature is particularly useful for parking lots or any environment requiring automated vehicle tracking.

### Connecting to an RTSP Stream

To connect to an RTSP stream from an IP camera, use the following command:

```bash
python live_detect.py --rtsp-url rtsp://username:password@camera_ip:port --conf-thres 0.5 --iou-thres 0.4
```

- `--rtsp-url`: The RTSP URL of the IP camera.
- `--conf-thres`: Confidence threshold for detections.
- `--iou-thres`: Intersection over Union threshold for non-max suppression.

### Using the Local Camera

To use the local camera for live detection, execute:

```bash
python live_detect.py --use-local-camera --conf-thres 0.5 --iou-thres 0.4
```

- `--use-local-camera`: Flag to indicate using the local camera instead of an RTSP stream.

## Database Integration

The project includes a fully integrated database solution for storing vehicle entry and exit times. The database automatically logs each detected license plate's entry time when it first enters the camera's view. If a detected plate already has an entry time but no exit time, the system logs the current time as the exit time.

### Database Schema

Here's a sample schema of the SQL database used in the project:

```sql
CREATE TABLE parking_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    license_plate TEXT NOT NULL,
    entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    exit_time TIMESTAMP
);
```

- `id`: Unique identifier for each record.
- `license_plate`: The recognized license plate number.
- `entry_time`: Timestamp of when the vehicle enters the parking lot.
- `exit_time`: Timestamp of when the vehicle exits the parking lot.

### Example Database View

| ID  | License Plate | Entry Time          | Exit Time          |
|-----|---------------|---------------------|--------------------|
| 1   | ABC1234       | 2024-07-24 09:15:00 | 2024-07-24 11:30:00|
| 2   | XYZ5678       | 2024-07-24 09:45:00 |                    |
| 3   | LMN8901       | 2024-07-24 10:00:00 | 2024-07-24 11:00:00|

### Accessing the Database

You can view or modify the database records using any SQL client or through a simple Python script like this:

```python
import sqlite3

def view_records():
    conn = sqlite3.connect('parking.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM parking_records")
    records = cursor.fetchall()
    for record in records:
        print(record)
    conn.close()

view_records()
```

## Example Outputs

Here are some examples of the system's outputs during operation:

### Example 1: Detected License Plate

**Input Frame:**

![Car Image](examples/example2.jpg)

### Example 2: Live Camera Feed

**Real-time Detection:**

![Live Detection](examples/live_cam_ss.png)

### Example 3: Database Entry

**Database Record:**

![Database Entry](examples/database.png)

## How It Works

The system uses a combination of deep learning, machine learning, and image processing techniques to detect, recognize, and log license plates.

### Workflow

1. **Capture Video Frame:** The system captures frames from the video stream in real-time using OpenCV.
2. **License Plate Detection:** The YOLOv8 model processes each frame to detect potential license plates.
3. **OCR Processing:** Detected plates are passed through an Optical Character Recognition (OCR) module to read the license plate numbers.
4. **Database Logging:** The recognized license plates are logged into the database with their entry and exit times.
5. **User Feedback:** Optionally, the system displays the detection results and database status on a user interface.

### Detailed Explanation

#### Step 1: Capture Video Frame

- **Camera Input:** The system connects to the camera using RTSP or local camera access, capturing live video frames.
- **Frame Processing:** Each frame is preprocessed for noise reduction and normalization.

#### Step 2: License Plate Detection

- **YOLOv8 Model:** The YOLOv8 model, trained on a dataset of car images, is used to detect bounding boxes around license plates.
- **Detection Thresholds:** Confidence and IoU thresholds are applied to filter out false positives.

#### Step 3: OCR Processing

- **Character Recognition:** Detected license plates are cropped from

 the image and processed through an OCR engine to extract text.
- **Text Validation:** Extracted text is validated against known formats to ensure accuracy.

#### Step 4: Database Logging

- **Entry Logging:** When a new license plate is detected, the entry time is logged in the database.
- **Exit Logging:** If a detected license plate already has an entry record, the exit time is updated in the database.

#### Step 5: User Feedback

- **UI Display:** An optional user interface displays real-time detection results, including bounding boxes and recognized text.
- **Database View:** The UI may also provide access to view and manage the database records.

### Architectural Flow

```plaintext
[Input Video Frame] --> [YOLOv8 Detection] --> [OCR Recognition] --> [Database Logging] --> [User Interface]
```

## Future Work

- **Integration with Payment Systems:** Expanding the system to integrate with automated payment systems for parking fees.
- **Improved OCR Accuracy:** Enhancing the OCR module with more robust character recognition techniques.
- **Multiple Camera Support:** Extending the system to support multiple camera streams simultaneously.
- **Advanced Analytics:** Implementing analytics features for traffic flow analysis and parking space management.
- **Cloud Deployment:** Exploring deployment options for cloud-based solutions, enabling remote monitoring and management.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions, comments, or suggestions, please reach out:

- **Name:** Görkem Turkut
- **Email:** gorkemturkut@hotmail.com
- **GitHub:** [gorkemturkut57](https://github.com/gorkemturkut57)

---
