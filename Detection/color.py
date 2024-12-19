import cv2 
import numpy as np

def detect_and_display_mask_center(frame):


  # Predefined color ranges for mask detection (adjust as needed)
  mask_color_ranges = {
    'red': (np.array([0, 100, 100]), np.array([10, 255, 255])),
    'green': (np.array([35, 100, 100]), np.array([75, 255, 255])),
    'blue': (np.array([110, 100, 100]), np.array([130, 255, 255])),
    'black': (np.array([0, 0, 0]), np.array([180, 255, 30])),
    'yellow': (np.array([20, 100, 100]), np.array([30, 255, 255])),
    'white': (np.array([0, 0, 221]), np.array([180, 30, 255])),  # Adjust for blue masks
  }

  # Convert the frame to HSV color space
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

  # Extract the central region of interest (ROI)
  center_x = frame.shape[1] // 2  # Integer division for center coordinates
  center_y = frame.shape[0] // 2
  roi_width = int(0.3 * frame.shape[1])  # Adjust ROI width as needed (e.g., 30% of frame width)
  roi_height = int(0.3 * frame.shape[0])  # Adjust ROI height as needed (e.g., 30% of frame height)
  roi_x1 = max(0, center_x - roi_width // 2)  # Ensure ROI stays within image bounds
  roi_y1 = max(0, center_y - roi_height // 2)
  roi = hsv[roi_y1:roi_y1+roi_height, roi_x1:roi_x1+roi_width]

  # Create masks for each color range and find dominant color
  mask_counts = {}
  for color, range in mask_color_ranges.items():
    mask = cv2.inRange(roi, range[0], range[1])
    mask_counts[color] = cv2.countNonZero(mask)  # Use cv2.countNonZero() for OpenCV 4+
  most_common_mask_color = max(mask_counts, key=mask_counts.get)

  # Display dominant color text (if a mask color is detected)
  if most_common_mask_color:
    if most_common_mask_color=="red":
        text = f"Trade Certificate : {most_common_mask_color}"
    elif most_common_mask_color=="green":
        text = f"Electric vehicle : {most_common_mask_color}"
    elif most_common_mask_color=="blue":
        text = f"Foreign Embassy vehicle : {most_common_mask_color}"
    elif most_common_mask_color=="black":
        text = f"Ministry of defence : {most_common_mask_color}"
    elif most_common_mask_color=="yellow":
        text = f"Transport vehicle : {most_common_mask_color}"
    elif most_common_mask_color=="white":
        text = f"Private ownership: {most_common_mask_color}"
  else:
    text = "No mask detected in center region"
  cv2.putText(frame, text, (20, 460), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

  # Display the processed frame
  cv2.imshow("color_img", frame)
  cv2.imshow("mask", mask)

# Initialize video capture
v = cv2.VideoCapture(0)

while True:
  # Read a frame from the video capture
  ret, frame = v.read()

  # If frame is read successfully, process and display
  if ret:
    detect_and_display_mask_center(frame.copy())  # Operate on a copy to avoid modifying original frame

    # Exit on 'q' key press
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
      break


# Release video capture and destroy windows
v.release()
cv2.destroyAllWindows()