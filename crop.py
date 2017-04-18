#!/usr/bin/env python
import cv2
import glob
num=0
image_list = []
for filename in glob.glob('amazontest10/*.jpg'): #assuming gif
    



 img = cv2.imread(filename)
 #crop_img = img[260:649, 477:1023] # Crop from x, y, w, h -> 100, 200, 300, 400
 crop_img = img[68:313, 448:810] #for book
 #[y1:y2, x1:x2]
 # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
 #cv2.imshow("cropped", crop_img)
 #cv2.waitKey(0)
 cv2.imwrite('amazontest10/pic'+str(num)+'.jpg', crop_img)
 num=num+1



