import cv2
import numpy as ny
face_cascade=cv2.CascadeClassifier('detect_face.xml')
cam= cv2.VideoCapture(0)              #('/media/inferni/Null/Movies/Ready Player One.mp4')
while True:
    img=cam.read()[1]
    graying=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(img,1.15,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        roi_gray=graying[y:y+h,x:x+w]
        roi_color=img[y:y+h,x:x+w]
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF==ord('z'):
        break
cv2.destroyAllWindows()
cam.release()
