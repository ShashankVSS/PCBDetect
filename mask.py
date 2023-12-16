import cv2
import numpy as np

# read the input as grayscale
img = cv2.imread('test1.jpeg', cv2.IMREAD_GRAYSCALE)
imgrgb = cv2.imread('test1.jpeg', cv2.IMREAD_COLOR)

# threshold
thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# get contours and filter out small defects
result = np.zeros_like(img)
contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


for cntr in contours:
    area = cv2.contourArea(cntr)
    if 10000000 > area > 1900000:
        cv2.drawContours(result, [cntr], 0, 255, thickness=cv2.FILLED)

# Create a binary mask from the drawn contours
mask = result.copy()

# Use the mask to extract the motherboard from the original rgb image
extracted_motherboard = cv2.bitwise_and(imgrgb, imgrgb, mask=mask)

# Save the extracted motherboard
cv2.imwrite('extracted_motherboard.png', extracted_motherboard)
cv2.imwrite('thresh.png', thresh)
cv2.imwrite('contours.png', result)

# resize results before display
result = cv2.resize(result, (0, 0), fx=0.2, fy=0.2)
thresh = cv2.resize(thresh, (0, 0), fx=0.2, fy=0.2)
extracted_motherboard = cv2.resize(extracted_motherboard, (0, 0), fx=0.2, fy=0.2)

# show results
cv2.imshow('thresh', thresh)
cv2.imshow('result', result)
cv2.imshow('extracted_motherboard', extracted_motherboard)
cv2.waitKey(0)


