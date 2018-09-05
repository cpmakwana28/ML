#!/usr/bin/python3
import cv2 as cv
import time
import os
cap = cv.VideoCapture(0)

while cap.isOpened():
	frame = cap.read()[1]
	cv.rectangle(frame,(200,100),(400,300),(0,0,255),2)
	frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
	cv.imshow("press q to quit,c to capture!  place your hands inside the rectangle",frame)	
	if cv.waitKey(1) ==  32:
		cv.imwrite("/home/cp28/Desktop/cap/capture.jpg",frame[100:300,200:400])
		time.sleep(1)
	if cv.waitKey(1) & 0xFF == ord('q'):
		cv.destroyAllWindows()
		cap.release()
		break

os.system('python3 label_image.py')
