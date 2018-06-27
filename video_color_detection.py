import cv2
import time
img=cv2.VideoCapture(0)
while img.isOpened():
    status,video=img.read()
    img1=cv2.inRange(video,(0,0,0),(50,50,255))
    #img2=cv2.bitwise_and(img,img1,mask=img1)
    cv2.imshow('img1',img1)
    #cv2.imshow('img2',img2)
    if cv2.waitKey(1) & 0xFF==ord('q') :
        break
cv2.destroyAllWindows()
img.release()