import cv2
import numpy as np
import pytesseract
import streamlit as st
from PIL import Image

# Function to detect if the given card is a PAN card
def is_pan_card(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    keywords = ['Income Tax Department', 'Permanent Account Number', 'INCOME TAX DEPARTMENT', 'GOVT. OF INDIA']
    for keyword in keywords:
        if keyword.lower() in text.lower():
            return True
    return False

# Function to detect tampering and calculate the percentage of tampered area
def detect_tampering(original_img, test_img):
    original_img_resized = cv2.resize(original_img, (test_img.shape[1], test_img.shape[0]))
    original_gray = cv2.cvtColor(original_img_resized, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    diff_img = cv2.absdiff(original_gray, test_gray)
    _, thresh_img = cv2.threshold(diff_img, 30, 255, cv2.THRESH_BINARY)
    tampered_area = np.sum(thresh_img > 0)
    total_area = thresh_img.size
    tampered_percentage = (tampered_area / total_area) * 100
    return tampered_percentage, thresh_img

# Function to analyze the card
def analyze_card(original_img, test_img):
    if not is_pan_card(test_img):
        return "The provided image is not a PAN card.", None

    tampered_percentage, tampered_img = detect_tampering(original_img, test_img)
    contours, _ = cv2.findContours(tampered_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    test_img_contours = test_img.copy()
    cv2.drawContours(test_img_contours, contours, -1, (0, 255, 0), 2)

    return f"The provided PAN card is tampered with {tampered_percentage:.2f}% of its area." if tampered_percentage > 0 else "The provided PAN card is not tampered.", test_img_contours

# Streamlit App Interface
st.title("PAN Card Tampering Detection")

# Upload the test image
test_img_file = st.file_uploader("Upload the Test PAN Card Image", type=["jpg", "jpeg", "png"])

if test_img_file is not None:
    test_img = np.array(Image.open(test_img_file))

    # Load the original PAN card image stored within the app
    original_img = cv2.imread('original_pan_card.jpg')  # Ensure this file is included in the repo

    # Analyze the card
    result, tampered_img = analyze_card(original_img, test_img)
    st.write(result)

    if tampered_img is not None:
        # Convert the tampered image to a PIL image
        tampered_img_pil = Image.fromarray(tampered_img)

        # Display the test image and the tampered detection result
        st.image([test_img, tampered_img_pil], caption=["Test PAN Card", "Tampered Detection"], use_column_width=True)
