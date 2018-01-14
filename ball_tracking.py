from collections import deque
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "yellow"
# ball in the HSV color space, then initialize the
# list of tracked pointsq
# greenLower = (29, 86, 6) # the lower and upper boundaries of the "green"
# greenUpper = (64, 255, 255) 
greenLower = (10, 100, 150)
greenUpper = (30, 255, 255)
# greenLower = (0, 0, 200)
# greenUpper = (100, 255, 255)
pts = deque(maxlen=args["buffer"])

# if a video path was not supplied, grab the reference
# to the webcamq
# if not args.get("video", False):
# 	camera = cv2.VideoCapture(0)
#
# # otherwise, grab a reference to the video file
# else:
namevideo = "Yellow_Ball_with_man.mp4"
camera = cv2.VideoCapture(namevideo)
if not camera.isOpened():
    print ("Error open video %s" %str(namevideo))
    exit()
# keep looping
while True:
	# grab the current frame
	(grabbed, frame) = camera.read()
	# Start timer
	timer = cv2.getTickCount()

	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if args.get("video") and not grabbed:
		break

	# resize the frame, blur it, and convert it to the HSV
	# color space
	#frame = imutils.resize(frame, width=1000)
	# blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask = cv2.inRange(hsv, greenLower, greenUpper)
	cv2.imshow('masdk', mask)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)
	cv2.imshow('mask', mask)
	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2]
	center = None
	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		# only proceed if the radius meets a minimum size
		if radius > 10:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			cv2.circle(frame, (int(x), int(y)), int(radius),
				(0, 255, 255), 2)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)

	#update the points queue
	pts.appendleft(center)

	#loop over the set of tracked points
	for i in xrange(1, len(pts)):
		# if either of the tracked points are None, ignore
		# them
		if pts[i - 1] is None or pts[i] is None:
			continue
		# otherwise, compute the thickness of the line and
		# draw the connecting lines
		thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
		cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
	cv2.putText(frame, "HSV Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
	#Calculate Frames per second (FPS)
	fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
	# Display FPS on frame
	cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
	#Display ball coodirate
	cv2.putText(frame, str(center), (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
	# show the frame to our screen
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	#if the space key is pressed, pause the loop
	if key == ord(" "):
		cv2.waitKey(0)
	# if the 'q' key is pressed, stop the loop
	if key == 27:
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
