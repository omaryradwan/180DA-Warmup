#This code is written by Omar Radwan, with samples and methods copied from
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html#converting-colorspaces

#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html#contour-features
#
#There was also the use of GIMP to figure out color schemes for the main objects of use(Surgical mask and latex gloves)
#There is also
#There was also the use of https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097 to figure out how to generate a color histogram, USE try.py for a simulation of this, not image.py
#
#
#
#
import cv2
import numpy as np
import dlib
import imutils
from imutils import face_utils
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

cap = cv2.VideoCapture(0)
def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist
def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart


while(1):

    _, frame = cap.read()

    lower_blue = np.array([89,80,80])
    upper_blue = np.array([107,255,255])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    res = cv2.bitwise_and(frame,frame, mask= mask)

    contours,hierarchy = cv2.findContours(mask, 1, 2)

    for i in contours:
        M = cv2.moments(i)
        #@print(M)
        area = cv2.contourArea(i)
        epsilon = 0.1*cv2.arcLength(i,True)
        approx = cv2.approxPolyDP(i,epsilon,True)
        #hull = cv2.convexHull(points[, hull[, clockwise[, returnPoints]]
        hull = cv2.convexHull

        x,y,w,h = cv2.boundingRect(i)
        #frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        if (w) > 60 and (h) > 60:
            res = cv2.rectangle(res,(x,y),(x+w,y+h),(0,255,0),2)
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)


    cv2.imshow('frame',frame)
    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
