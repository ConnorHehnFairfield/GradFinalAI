import cv2
import pytesseract
import csv

# Set the path to the Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\johnp\OneDrive\Desktop\pt\tesseract.exe'  # Update this path accordingly
import cv2
import pytesseract
from PIL import Image
import csv
import os

def preprocess_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to get a binary image
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)

    return binary_image

def extract_license_plate_number(image_path):
    # Perform OCR using Tesseract on the preprocessed image
    extracted_text = pytesseract.image_to_string(Image.fromarray(image_path), lang = 'eng', config='--psm 10')

    return extracted_text.strip()

def process_license_plates(image_folder, output_csv):
    # Create or open a CSV file to store license plate numbers
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Image', 'License Plate Number'])

        # Process each image in the specified folder
        for filename in os.listdir(image_folder):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(image_folder, filename)

                # Preprocess the image
                preprocessed_image = preprocess_image(image_path)

                # Extract license plate number
                license_plate_number = extract_license_plate_number(preprocessed_image)

                # Write the result to the CSV file
                csv_writer.writerow([filename, license_plate_number])

                print(f"Processed {filename}: {license_plate_number}")

if __name__ == "__main__":
    # Set the path to your folder containing license plate images
    input_image_folder = 'C:/Users/johnp/OneDrive/Desktop/AA Masters/Sem 3/Artificial Intelligence/Final/license_plate_final/photos'

    # Set the path for the output CSV file
    output_csv_file = 'license_plate_numbers.csv'

    # Process license plates and store results in CSV
    process_license_plates(input_image_folder, output_csv_file)
