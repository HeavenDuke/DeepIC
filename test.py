import cv2


def removeBackground(img1, img2):
    ret, mask = cv2.threshold(img2, 10, 255, cv2.THRESH_BINARY)
    contours, hiearchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print contours
    x, y, w, h = cv2.boundingRect(contours[0])
    # img1_bg = cv2.bitwise_and(img1, img1, mask = mask)
    return img1[y:y + h, x:x + w]

i1 = cv2.imread("./data/train/image/1.jpg", cv2.IMREAD_COLOR)
i2 = cv2.imread("./data/train/mask/1.png", cv2.IMREAD_GRAYSCALE)

print removeBackground(i1, i2).shape
