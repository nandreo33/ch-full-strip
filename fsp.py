import cv2
import numpy as np
import glob
import argparse

# call this function with command line:
#	 'python fsp.py -t "_path_/_to_/_file_.jpg"'
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--template", required=True)
args = vars(ap.parse_args())

template = cv2.imread(args["template"])
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# display the template pre-modification
cv2.imshow("iteration", template)
cv2.waitKey(0)
cv2.destroyAllWindows()

(tH, tW) = template.shape[:2]
max_height = 180;

# scale the template down
if max_height < tH:
	scaling_factor = max_height / float(tH);
	template = cv2.resize(template, None, fx=scaling_factor, fy=scaling_factor, interpolation = cv2.INTER_AREA)
	(tH, tW) = template.shape[:2]

# apply canny-edge detector for better matching
template = cv2.Canny(template,100,200)

# display GRAY-ed, scaled-down, Canny-ed template
cv2.imshow("iteration", template)
cv2.waitKey(0)
cv2.destroyAllWindows()

found = None
best = None
filename = None

# completion_percent = 0

# total_files = len(glob.glob("ch/*/*/*.jpg"));
# files_completed = 0

# to search all files:
#		for imagePath in glob.glob("ch/*/*/*.jpg"):

for imagePath in glob.glob("ch/1990/*/*.jpg"):
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	for scale in np.linspace(0.2,1.2,21)[::-1]:
		resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation = cv2.INTER_AREA)
		r = gray.shape[1] / float(resized.shape[1])
		name = imagePath

		if resized.shape[0] < tH or resized.shape[1] < tW:
			break

		edged = cv2.Canny(resized,100,200)
		result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
		(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

		if found is None or maxVal > found[0]:
			found = (maxVal, maxLoc, r)
			best = image
			filename = name

	# files_completed += 1
	# if (files_completed*100/total_files)%10 == 0:
	# 	print(str(files_completed*100/total_files) + "% complete")

(_, maxLoc, r) = found
(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

# display best match with overlaid red rectangle
cv2.rectangle(best, (startX, startY), (endX, endY), (0,0,255), 2)
cv2.imshow(name, best)
cv2.waitKey(0)
cv2.destroyAllWindows()