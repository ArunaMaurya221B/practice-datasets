import cv2
import numpy as np



#reading the image
img = cv2.imread('/home/icts/practice-datasets/Electric-Meters/Electric-meters/MAN_5001816631_20170819_OK.jpg')

#median blur to remove the noise and other small characters
median = cv2.medianBlur(img, 3)

#converting to hsv
hsv = cv2.cvtColor(median, cv2.COLOR_BGR2HSV)

#defining the green mask and extracting the screen
mask = cv2.inRange(hsv, (65,60,60), (80, 255,255))
res = cv2.bitwise_and(img, img, mask=mask)

#converting to grayscale and thresholding
gray_img = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
flag, thresh = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY_INV)

#doing the final morphological transformations
kernel = np.ones((3,3), np.uint8)
erosion = cv2.erode(thresh,kernel,iterations = 1)
dilation = cv2.dilate(erosion,kernel,iterations = 4)
final_image = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel)
#cv2.imshow("Image", final_image)


#using canny edge detector to detect edges
edges = cv2.Canny(final_image, 0, 200, 255)
#cv2.imshow("Image", edges)

#finding contours
im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
idx=0
final_contours = []

#final computation of bounding rects and calulation of RoI's
for c in contours:
	x,y,w,h = cv2.boundingRect(c)
	if h>55 and w<64:
		cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0),2)
		cv2.imshow("Image", img)
		final_contours.append(c)
		roi=img[y:y+h, x:x+w]
		cv2.imwrite(str(idx) + '.jpg', roi)
		idx=idx+1

cv2.waitKey(0)
cv2.destroyAllWindows()