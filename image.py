
import cv2

img_file = "./zidane.jpg" 
img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)    # 회색으로 읽기

if img is not None:
  cv2.imshow('IMG', img)
  cv2.waitKey()
  cv2.destroyAllWindows()
else:
    print('No image file.')
    