import cv2
import numpy as np




i1 = cv2.imread("./data/train/image/1.jpg", cv2.IMREAD_COLOR)
i2 = cv2.imread("./data/train/mask/1.png", cv2.IMREAD_GRAYSCALE)

cv2.imshow("image", removeBackground(i1, i2))
cv2.waitKey()
cv2.destroyAllWindows()