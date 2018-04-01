import cv2
import numpy as np
import time

#Makes a color gradient to tell which points are which.
colors = np.zeros((1, 6, 3), np.uint8)
colors[0,:] = 255
colors[0,:,0] = np.arange(0, 180, 180.0/6)
colors = cv2.cvtColor(colors, cv2.COLOR_HSV2BGR)[0]

#Read in image
img = cv2.imread('d4.png', 1)
start = time.time()
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#Conver to HSV to filter better
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
#Add adaptive threshold to make sure it finds the lighter pips
'''
adt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 1)
for i in range(img.shape[0]):
	for j in range(img.shape[1]):
		hsv[i,j,2] = (45*hsv[i,j,2] + 65*adt[i,j])/100
'''

#Filters out the darker pixels by value
mask = cv2.inRange(hsv, np.array([0,0,30]), np.array([127,30,255]))
#Blurs the dice in a way that the smaller pips get blurred out.
element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6))
mask1 = cv2.GaussianBlur(mask,(5, 5), 3)
mask2 = cv2.dilate(mask1, element, iterations = 1)
mask3 = cv2.erode(mask2, element, iterations = 1)
horiz_concat = np.concatenate((mask1,mask2,mask3), axis=1)
mask = mask3

ret, mask = cv2.threshold(mask,100,255,0)
#Why do you have to add 2? I don't know, but it works.
shitMask = np.zeros((mask.shape[0]+2,mask.shape[1]+2),np.uint8)
#Starts at corner, and makes everything in the background white, starting there.
num,im,mask,rect = cv2.floodFill(mask, shitMask, (0,0), (255,255,255))

#cv2.imshow('Other', horiz_concat)

#Finds contours
_, contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
numberOfDice = 6

#Delete bad contour you get from shitMask
contours = np.delete(contours,0,0)
#Draw the contours.
im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
cv2.drawContours(im,contours,-1,(0,255,0),3)
cx = []
cy = []
#Find the moments/centers
for contour in contours:
	M = cv2.moments(contour)
	cx.append(int(M['m10']/M['m00']))
	cy.append(int(M['m01']/M['m00']))
points = zip(cx,cy)
z = np.float32(points)

#K means clustering for 6 dice	
compactness, labels, centers = cv2.kmeans(z, 6, None, (cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)\

colorArr = np.zeros(20)
distanceArr = np.zeros(20)
# Each entry is a list of four values for a bounding box, xMin, xMax, yMin, yMax
imgBoxArray = np.zeros((6,4))
for arr in imgBoxArray:
	arr[2] = 99999
	arr[0] = 99999
'''diceArr = np.zeros(7)'''
#Draw points
for (x, y), label in zip(np.int32(z), labels.ravel()):
	c = list(map(int, colors[label]))
	cv2.circle(img, (x, y), 3, c, -1)
	colorArr[label] += 1
	distanceArr[label] += (centers[label,0] - x)**2 + (centers[label,1] - y)**2
	if(x < imgBoxArray[label, 0]):
		imgBoxArray[label, 0] = x
	if(x > imgBoxArray[label, 1]):
		imgBoxArray[label, 1] = x
	if(y < imgBoxArray[label, 2]):
		imgBoxArray[label, 2] = y
	if(y > imgBoxArray[label, 3]):
		imgBoxArray[label, 3] = y

updateArr = np.copy(colorArr)

'''for i in range(len(colorArr)):
	if colorArr[i] > 7:
		colorArr[i]/=2
	diceArr[int(colorArr[i])] += 1

for i in range(len(colorArr)):
	if(diceArr[int(colorArr[i])] >= 2):
		minDist = 9999999999.999
		for j in range(len(diceArr)):
'''
for arr in imgBoxArray:
	if(arr[0]==arr[1]):
		arr[0]-=15
		arr[1]+=15
	if(arr[2]==arr[3]):
		arr[2]-=15
		arr[3]+=15
	arr[0]-=40
	arr[1]+=40
	arr[2]-=20
	arr[3]+=20
	cv2.rectangle(img, (int(arr[0]),int(arr[2])),(int(arr[1]),int(arr[3])),(255,0,0),2)
end = time.time()
print(end-start)

#Draw number of pips
for (x, y), label in zip(np.int32(z), labels.ravel()):
	c = colors[label].tostring()
	cv2.putText(img, str(int(colorArr[label])), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255))

cv2.imshow('die2', img)

while (cv2.waitKey(1)&0xFF) != 27:
	pass

cv2.destroyAllWindows()
