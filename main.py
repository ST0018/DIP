import cv2
import numpy as np
import pytesseract
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Opening an image
img = cv2.imread('last.jpg')

# Convert to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray_img = 0.2989 * img[::3] + 0.5870 * img[::2] + 0.1140 * img[::1]

# Gamma correction
gamma = 1.5
gam_Tx_img = np.array(255 * (gray_img / 255) ** gamma, dtype='uint8')
cv2.imshow('gamma_corr.jpg', gam_Tx_img)
cv2.waitKey(0)
cv2.imwrite('Gamma_Corrected_Img.jpg', gam_Tx_img)

# Complementary image
comp_img = 255 - gray_img
cv2.imshow('Compl_Img.jpg', comp_img)
cv2.waitKey(0)
cv2.imwrite('Complementary_Img.jpg', comp_img)


# Find frequency of pixels in range 0-255
histr_bfr = cv2.calcHist([comp_img], [0], None, [256], [0, 256])

# Show the plotting graph of an image
plt.plot(histr_bfr)
plt.show()

# Binarization - Otsu's Thresholding

#_, binary_img = cv2.threshold(comp_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#cv2.imshow('Final_image.jpg', binary_img)
#cv2.waitKey(0)
#cv2.imwrite('Final_Img.jpg', binary_img)
# Save the binarized image to a file for OCR
#cv2.imwrite('Binarized_Image_For_OCR.jpg', binary_img)

# Calculate number of bins
bins_num = 256

# Get the image histogram
hist, bin_edges = np.histogram (comp_img, bins=bins_num)

# Calculate centers of bins
bin_mids = (bin_edges[:-1]+ bin_edges [1:]) / 2.

# Iterate over all thresholds (indices) and get the probabilities wi(t), w2(t)
weight1 = np.cumsum(hist)
weight2 = np.cumsum(hist[::-1])[::-1]

# Get the class means mu8(t)
mean1 = np.cumsum(hist * bin_mids) / weight1

# Get the class means mu1(t)
mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]

inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
index_of_max_val = np.argmax(inter_class_variance)


# Maximize the inter_class_variance function val
threshold = bin_mids[:-1][index_of_max_val]


print("Otsu's algorithm implementation thresholding result: ", threshold)

h = comp_img.shape[0]
w = comp_img.shape[1]

for i in range(h):
    for j in range(w):
        if comp_img[i, j] < threshold:
            comp_img[i, j] = 0
        else:
            comp_img[i, j] = 255

cv2.imshow('Thresholded Image', comp_img)
cv2.waitKey(0)

# Save the binarized image to a file for OCR
cv2.imwrite('Binarized_Image_For_OCR.jpg', comp_img)

# Find frequency of pixels in range 0-255
histr_aft = cv2.calcHist([comp_img], [0], None, [256], [0, 256])

# Show the plotting graph of an image
plt.plot(histr_aft)
plt.show()

# Perform OCR on the binarized image
text = pytesseract.image_to_string('Binarized_Image_For_OCR.jpg')
# print("Extracted Text:", text)

# Save the text as an XML file
root = ET.Element("document")
text_element = ET.SubElement(root, "text")
text_element.text = text

tree = ET.ElementTree(root)
xml_file_path = 'output.xml'
tree.write(xml_file_path)


# Accessing elements in the XML
text_element = root.find('text')
if text_element is not None:
    extracted_text = text_element.text
    print(f'Extracted Text from XML: {extracted_text}')
else:
    print('No "text" element found in the XML.')
