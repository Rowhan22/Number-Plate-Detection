import cv2
import numpy as np

# Predefined color ranges for mask detection (adjust as needed)
mask_color_ranges = {
    'red': (np.array([0, 100, 100]), np.array([10, 255, 255])),
    'green': (np.array([35, 100, 100]), np.array([75, 255, 255])),
    'blue': (np.array([110, 100, 100]), np.array([130, 255, 255])),
    'black': (np.array([0, 0, 0]), np.array([180, 255, 30])),
    'yellow': (np.array([20, 100, 100]), np.array([30, 255, 255])),
    'white': (np.array([0, 0, 221]), np.array([180, 30, 255])),  # Adjust for blue masks
}

# Initialize video capture
cap = cv2.VideoCapture(0)

# Load the pre-trained cascade classifier for license plate detection
plateCascade = cv2.CascadeClassifier("number_plate_detection.xml")

# Minimum area threshold for detecting license plates
minArea = 500

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # If frame is read successfully, process and display
    if ret:
        # Convert frame to grayscale
        imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect license plates in the frame
        numberPlates = plateCascade.detectMultiScale(imgGray, 1.1, 4)

        for (x, y, w, h) in numberPlates:
            area = w * h
            if area > minArea:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, "NumberPlate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                imgRoi = frame[y:y + h, x:x + w]

                # Convert ROI to HSV color space
                imgRoiHSV = cv2.cvtColor(imgRoi, cv2.COLOR_BGR2HSV)

                # Find the dominant color of the license plate
                mask_counts = {}
                for color, range in mask_color_ranges.items():
                    mask = cv2.inRange(imgRoiHSV, range[0], range[1])
                    mask_counts[color] = cv2.countNonZero(mask)  # Use cv2.countNonZero() for OpenCV 4+
                most_common_mask_color = max(mask_counts, key=mask_counts.get)

                # Display vehicle type based on the color of the license plate
                if most_common_mask_color:
                    if most_common_mask_color == "red":
                        vehicle_type = "Trade Certificate"
                    elif most_common_mask_color == "green":
                        vehicle_type = "Electric vehicle"
                    elif most_common_mask_color == "blue":
                        vehicle_type = "Foreign Embassy vehicle"
                    elif most_common_mask_color == "yellow":
                        vehicle_type = "Transport vehicle"
                    elif most_common_mask_color == "white":
                        vehicle_type = "Private ownership"
                else:
                    vehicle_type = "Unknown"

                # Display vehicle type
                cv2.putText(frame, f"Vehicle Type: {vehicle_type}", (x, y - 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("Number Plate", imgRoi)

        cv2.imshow("Result", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and destroy windows
cap.release()
cv2.destroyAllWindows()
