from scipy.spatial import distance as dist
import imutils
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import cv2


"""
scattering을 하기 위해서는 무엇을 해야할까?

Scenario
1. 물건 두 개가 겹쳐 있다.
2. 물건 두 개가 옆에 나란히 있다.

Solution for no.1
1. 가장 위에 있는 object를 집는다.
2. 아무 물건이 없는 곳에 치운다.

Solution for no.2
1. 두 물건 사이에 있는 틈으로 gripper를 close한 채로 밀어넣는다.
2. 옆에 있는 물건을 옆으로 친다, 또는 밀어낸다.

Logistics
1. Get neightboring objects from one object
    1. select one object.
    2. get distance from other objects
    3. if distance is small, then that is neighboring object
2. 
"""

def get_neighbor_obj():
    # 어떻게 detect 해야할까?
    # object의 경계선은 어디에 있지?
    return [1, 2, 3]

def put_neighbor_obj(obj):
    # neighboring object 옮기기
    # 어떻게 옮기지?
    pass

def scatter():
    neighbor_obj_list = get_neighbor_obj()
    for neighbor_obj in neighbor_obj_list:
        put_neighbor_obj(neighbor_obj)

# original contents from:
# https://www.pyimagesearch.com/2016/04/04/measuring-distance-between-objects-in-an-image-with-opencv/
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def get_distance(obj_1, obj_2):
    """
    Get the distance between object1 and object2.
    TODO:
    1. Split the function to object detector and distance computer.
    2. Specify the real distance between two objects.
    3. REMOVE INCHES.
    """
    # load the image, convert it to grayscale, and blur it slightly
    image = cv2.imread("C:/Users/incorl_robot/Desktop/2020PartTimeJob/Robot-Runner/Robot_env/test_image.png")    # In this case, it should be image from
                                            # Intel realsense camera.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # sort the contours from left-to-right and, then initialize the
    # distance colors and reference object
    (cnts, _) = contours.sort_contours(cnts)
    colors = ((0, 0, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0),
        (255, 0, 255))
    refObj = None

    # loop over the contours individually
    for c in cnts:
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < 100:    # This should be changed with relate to
            continue                    # robot environment.
        # compute the rotated bounding box of the contour
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        # order the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left
        # order, then draw the outline of the rotated bounding
        # box
        box = perspective.order_points(box)
        # compute the center of the bounding box
        cX = np.average(box[:, 0])
        cY = np.average(box[:, 1])

        # if this is the first contour we are examining (i.e.,
        # the left-most contour), we presume this is the
        # reference object
        if refObj is None:
            # unpack the ordered bounding box, then compute the
            # midpoint between the top-left and top-right points,
            # followed by the midpoint between the top-right and
            # bottom-right
            (tl, tr, br, bl) = box
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)
            # compute the Euclidean distance between the midpoints,
            # then construct the reference object
            D = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
            refObj = (box, (cX, cY), D / 600)   # 600 is the width of test_image.png
            continue

        # draw the contours on the image
        orig = image.copy()
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
        cv2.drawContours(orig, [refObj[0].astype("int")], -1, (0, 255, 0), 2)
        # stack the reference coordinates and the object coordinates
        # to include the object center
        refCoords = np.vstack([refObj[0], refObj[1]])
        objCoords = np.vstack([box, (cX, cY)])

        # TODO: getting the nearest distance between two objects
        # loop over the original points
        for ((xA, yA), (xB, yB), color) in zip(refCoords, objCoords, colors):
            # draw circles corresponding to the current points and
            # connect them with a line
            cv2.circle(orig, (int(xA), int(yA)), 5, color, -1)
            cv2.circle(orig, (int(xB), int(yB)), 5, color, -1)
            cv2.line(orig, (int(xA), int(yA)), (int(xB), int(yB)),
                color, 2)
            # compute the Euclidean distance between the coordinates,
            # and then convert the distance in pixels to distance in
            # units
            D = dist.euclidean((xA, yA), (xB, yB)) / refObj[2]
            (mX, mY) = midpoint((xA, yA), (xB, yB))
            cv2.putText(orig, "{:.1f}in".format(D), (int(mX), int(mY - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            # show the output image
            cv2.imshow("Image", orig)
            cv2.waitKey(0)

if __name__ == "__main__":
    get_distance(1, 2)