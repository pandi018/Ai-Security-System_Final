import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# initializing subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

while(True):
	ret, frame = cap.read()

	# applying on each frame
	fgmask = fgbg.apply(frame)
	re_mask = cv2.resize(fgmask, (500, 500))
	frame=cv2.resize(frame,(500,500))
	cv2.imshow("Frame", frame)
	cv2.imshow("sub_frame",re_mask)
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

# import numpy as np
# import cv2
#
# cap = cv2.VideoCapture(0)
#
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#
# # initializing subtractor
# fgbg = cv2.createBackgroundSubtractorMOG2()
#
# while (1):
# 	ret, frame = cap.read()
#
# 	# applying on each frame
# 	fgmask = fgbg.apply(frame)
#
# 	fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
# 	re_mask=cv2.resize(fgmask,(500,500))
# 	frame=cv2.resize(frame,(500,500))
# 	# imS = cv2.resize(im, (960, 540))
#
# 	cv2.imshow('re_frame', re_mask)
# 	cv2.imshow("frame",frame)
# 	k = cv2.waitKey(1) & 0xff
# 	if k == ord('q'):
# 		break
#
# cap.release()
# cv2.destroyAllWindows()