import cv2
import numpy as np
import imutils
import easyocr
from matplotlib import pyplot as plt

# Input 
path = r"C:\Users\hp\Desktop\ANPR MODEL\tesla.jpg"
img = cv2.imread(path, 1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0)

plt.imshow(gray, cmap='gray')
plt.title('Car')
plt.show()

# Noise Reduction and Edge Detection
bfilter = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(bfilter, 30, 200)

plt.imshow(edged, cmap='gray')
plt.title('Edge Detection')
plt.show()

# Shape Detection / Contour Detection
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break

print(location)

# Masking an Image
mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0, 255, -1)
new_image = cv2.bitwise_and(img, img, mask=mask)

plt.imshow(new_image)
plt.title('Masked Image')
plt.show()

(x, y) = np.where(mask == 255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2 + 1, y1:y2 + 1]

plt.imshow(cropped_image, cmap='gray')
plt.title('Cropped Image')
plt.show()

# Reading of Masked Image/EasyOCR
reader = easyocr.Reader(['en'])
result = reader.readtext(cropped_image)
print(result)

# Render
text = result[0][-2]
font = cv2.FONT_HERSHEY_SIMPLEX
res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1] + 60), fontFace=font, fontScale=1,
                  color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 3)

plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
plt.title('Result')
plt.show()
