import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
from PIL import Image
import easyocr
import numpy as np
import re
import sqlite3
from datetime import datetime, timedelta

# Establish connection to SQLite database
db_file = 'license_plates.db'
conn = sqlite3.connect(db_file)
c = conn.cursor()

# Create table if not exists
c.execute('''
CREATE TABLE IF NOT EXISTS plates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate TEXT,
    entry_time TEXT,
    exit_time TEXT,
    UNIQUE(plate, entry_time)
)
''')
conn.commit()

# Load the model
model_path = 'best.pt'
model = YOLO(model_path)

# Initialize the EasyOCR reader
reader = easyocr.Reader(['tr'])

# Function to process images
def process_image(img):
    # Perform prediction using the model
    results = model(img)

    for result in results:
        boxes = result.boxes.xyxy
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            # Draw the detected box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Increase thickness

            # Crop the box and perform OCR
            img_pil = Image.fromarray(img)
            cropped_image = img_pil.crop((x1, y1, x2, y2))
            image_np = np.array(cropped_image)

            # Perform OCR
            result = reader.readtext(image_np)

            result_text = ""
            max_confidence = 0.0
            for detection in result:
                text = detection[1]
                confidence = detection[2]
                result_text += text  # Concatenate texts
                if confidence > max_confidence:
                    max_confidence = confidence

            # Remove spaces and convert to uppercase
            result_text = result_text.replace(" ", "").upper()
            result_text = result_text.replace("-", "")

            # Search for Turkish plate format
            match = re.search(r'\d{2}[A-Z]{1,3}\d{2,4}', result_text)

            if match and max_confidence > 0.8:
                plaka = match.group(0)
                accuracy = int(max_confidence * 100)  # Convert to percentage

                # Add the plate to the image
                display_text = f"{plaka} %{accuracy}"
                cv2.putText(img, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)  # font size and thickness

                # Check the plate in the database
                c.execute("SELECT * FROM plates WHERE plate=? ORDER BY entry_time DESC LIMIT 1", (plaka,))
                plate_record = c.fetchone()

                current_time = datetime.now()
                current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")

                if plate_record is None:
                    # New plate, record entry time
                    c.execute("INSERT INTO plates (plate, entry_time, exit_time) VALUES (?, ?, NULL)", (plaka, current_time_str))
                    print(f"New plate detected: {plaka}, entry time: {current_time_str}")
                else:
                    last_entry_time = datetime.strptime(plate_record[2], "%Y-%m-%d %H:%M:%S")
                    if plate_record[3] is None:
                        # Exit time not yet recorded
                        if current_time - last_entry_time > timedelta(minutes=2):
                            c.execute("UPDATE plates SET exit_time=? WHERE plate=?", (current_time_str, plaka))
                            print(f"Plate exited: {plaka}, exit time: {current_time_str}")
                        else:
                            print(f"Plate {plaka} exit ignored due to 2-minute rule.")
                    else:
                        # Exit time recorded, check for new entry
                        last_exit_time = datetime.strptime(plate_record[3], "%Y-%m-%d %H:%M:%S")
                        if current_time - last_exit_time > timedelta(minutes=2):
                            c.execute("INSERT INTO plates (plate, entry_time, exit_time) VALUES (?, ?, NULL)", (plaka, current_time_str))
                            print(f"Plate re-entered: {plaka}, entry time: {current_time_str}")
                        else:
                            print(f"Plate {plaka} entry ignored due to 2-minute rule.")

                conn.commit()

            else:
                cv2.putText(img, "Plate Not Detected!", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)  # Increase font size and thickness

    return img

# Function to process videos
def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Start video writer
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process each frame
        processed_frame = process_image(frame)
        out.write(processed_frame)

    # Release video resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Output video saved to {output_path}")

def main(input_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process input file based on its type
    if input_path.endswith(('.jpg', '.jpeg', '.png')):
        img = cv2.imread(input_path)
        output_path = os.path.join(output_dir, 'result.jpg')
        processed_img = process_image(img)
        cv2.imwrite(output_path, processed_img)
        plt.imshow(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
        # plt.show()
    elif input_path.endswith(('.mp4', '.avi', '.mov')):
        output_path = os.path.join(output_dir, 'result_video.mp4')
        process_video(input_path, output_path)
    elif input_path == 'camera':
        # Capture from camera
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process each frame
            processed_frame = process_image(frame)

            # Show the processed frame
            cv2.imshow('Live Feed', processed_frame)

            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        print("Unsupported input type.")

# Example usage for camera capture
input_path = 'testing_images/1.jpg'
output_dir = 'output_directory'
main(input_path, output_dir)