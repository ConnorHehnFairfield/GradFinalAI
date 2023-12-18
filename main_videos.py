import cv2
import pytesseract
import csv
import numpy as np

# Set the path to the Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\johnp\OneDrive\Desktop\pt\tesseract.exe'  # Update this path accordingly

def is_valid_license_plate(rect):
    # Modify these thresholds based on your license plate characteristics
    min_ratio = 2.5  # Minimum bounding rectangle sides ratio
    max_ratio = 4.0  # Maximum bounding rectangle sides ratio
    min_area = 800   # Minimum bounding rectangle area

    # Calculate width, height, and area of the bounding rectangle
    _, _, w, h = rect
    area = w * h

    # Calculate the sides ratio
    ratio = max(w / h, h / w)

    # Check if the bounding rectangle meets the criteria
    return min_ratio < ratio < max_ratio and area > min_area

# Open the video file
video_path = 'C:/Users/johnp/OneDrive/Desktop/AA Masters/Sem 3/Artificial Intelligence/Final/license_plate_final/Videos/video.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video file is opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Create a CSV file to write license plate information
csv_file_path = 'license_plate_data.csv'
with open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Frame', 'License Plate'])

    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Additional preprocessing steps
        resized_frame = cv2.resize(frame, (0, 0), fx=2, fy=2)  # Adjust the scaling factor as needed
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        _, binary_frame = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        denoised_frame = cv2.fastNlMeansDenoising(binary_frame, None, 10, 7, 21)

        # Debugging OCR output
        original_ocr_output = pytesseract.image_to_string(denoised_frame, config=r'--oem 3 --psm 6 outputbase digits')
        print(f"Frame {frame_number} Original OCR Output:", original_ocr_output)

        # Define a dictionary to map common misread characters
        character_mapping = {'6': 'G', '8': 'B'}

        # Replace misread characters in the OCR output
        refined_ocr_output = ''.join(character_mapping.get(char, char) for char in original_ocr_output)

        # Debugging OCR output after refinement
        print(f"Frame {frame_number} Refined OCR Output: {refined_ocr_output}")

        # Write frame number and refined license plate to CSV
        csv_writer.writerow([frame_number, refined_ocr_output.strip()])

        # Display the frame with denoised license plate region
        cv2.imshow('License Plate', denoised_frame)

        # Press 'q' to exit the video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_number += 1

    # Release the video capture object and close CSV file
    cap.release()
    cv2.destroyAllWindows()

print(f"License plate information has been extracted and saved to {csv_file_path}.")
