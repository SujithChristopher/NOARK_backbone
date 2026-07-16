import cv2
import numpy as np
img = np.zeros((400,400), dtype=np.uint8)
cv2.imshow("Test", img)
cv2.waitKey(0) # Press any key to close