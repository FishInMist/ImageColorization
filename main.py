# Author: xwang875

import cv2
from Colorization import colorization


# cutoffMode = 0: my implementation; cutoffMode = 1: source code implementation

# Original paper replication
originalImage0 = cv2.imread("./example.bmp")
markedImage0 = cv2.imread("./example_marked.bmp")
colorization(originalImage0, markedImage0, "example", cutoffMode=1)
print "Result 0 is finished"

# My result 1
originalImage1 = cv2.imread("./potatoBW_50.bmp")
markedImage1 = cv2.imread("./potatoBW_50M.bmp")
colorization(originalImage1, markedImage1, "Result1", cutoffMode=0)
print "Result 1 is finished"

# My result 2
originalImage2 = cv2.imread("./ballonDog_50.bmp")
markedImage2 = cv2.imread("./ballonDog_50M.bmp")
colorization(originalImage2, markedImage2, "Result2", cutoffMode=1)
print "Result 2 is finished"

# My result 3
originalImage3 = cv2.imread("./Cat_Turkey_50.bmp")
markedImage3 = cv2.imread("./Cat_Turkey_50M.bmp")
colorization(originalImage3, markedImage3, "Result3", cutoffMode=1)
print "Result 3 is finished"
