import cv2
import numpy as np
import pytesseract
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

# Function to detect if the given card is a PAN card
def is_pan_card(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use OCR to extract text
    text = pytesseract.image_to_string(gray)

    # Check for specific keywords in the text
    keywords = ['Income Tax Department', 'Permanent Account Number', 'INCOME TAX DEPARTMENT', 'GOVT. OF INDIA']
    for keyword in keywords:
        if keyword.lower() in text.lower():
            return True
    return False

# Function to detect tampering and calculate the percentage of tampered area
def detect_tampering(original_img, test_img):
    # Resize images to the same dimensions
    original_img_resized = cv2.resize(original_img, (test_img.shape[1], test_img.shape[0]))

    # Convert images to grayscale
    original_gray = cv2.cvtColor(original_img_resized, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    # Compute the absolute difference between the images
    diff_img = cv2.absdiff(original_gray, test_gray)

    # Threshold the difference image to get the regions with significant changes
    _, thresh_img = cv2.threshold(diff_img, 30, 255, cv2.THRESH_BINARY)

    # Calculate the percentage of tampered area
    tampered_area = np.sum(thresh_img > 0)
    total_area = thresh_img.size
    tampered_percentage = (tampered_area / total_area) * 100

    return tampered_percentage, thresh_img

# Function to analyze the card
def analyze_card(original_img, test_img):
    if not is_pan_card(test_img):
        return "The provided image is not a PAN card.", None

    tampered_percentage, tampered_img = detect_tampering(original_img, test_img)

    # Convert OpenCV image to PIL Image
    tampered_img_pil = Image.fromarray(cv2.cvtColor(tampered_img, cv2.COLOR_GRAY2RGB))
    original_img_pil = Image.fromarray(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    test_img_pil = Image.fromarray(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))

    if tampered_percentage > 0:
        return f"The provided PAN card is tampered with {tampered_percentage:.2f}% of its area.", [original_img_pil, test_img_pil, tampered_img_pil]
    else:
        return "The provided PAN card is not tampered.", [original_img_pil, test_img_pil, tampered_img_pil]

# Streamlit App
st.title('PAN Card Tampering Detection')

# Load the images
original_img = cv2.imread('original_pan_card.jpg')  # Ensure the path is correct
test_img = cv2.imread('test_pan_card.jpg')  # Ensure the path is correct

if original_img is not None and test_img is not None:
    result, images = analyze_card(original_img, test_img)
    st.write(result)

    if images:
        st.image(images, caption=["Original PAN Card", "Test PAN Card", "Tampered Detection"], use_column_width=True)
else:
    st.write("Please upload the images correctly.")
