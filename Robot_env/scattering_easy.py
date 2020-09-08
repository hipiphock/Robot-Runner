from scipy.spatial import distance as dist
import imutils
from imutils import contours
import numpy as np
import cv2
import copy
import logging
from Robot_env.config import RL_Obj_List

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

scattering_logger = logging.getLogger("scattering_easy")


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


def get_distance(image, robot, detected_obj_lists):
    """
    Get the distance between objects
    TODO:
    1. Split the function to object detector and distance computer.
    2. Specify the real distance between two objects.
    3. Identify objects! (IMPORTANT)
    """
    # distance_array is an array that represents distance between two objects.
    # For example, if cup_red(9) and big_box(12)'s distance is 11.82, then distance_aray[9][12] = 11.82.
    distance_array = np.ones((50, 50))
    distance_array = np.full_like(distance_array, 999.99)

    # load the image, convert it to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (7, 7), 0)
    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    # TODO: 기존 코드에서 object 경계 가져올 것 ?
    edged = cv2.Canny(gray, 50, 100)
    # edged = cv2.dilate(edged, None, iterations=1)
    # edged = cv2.erode(edged, None, iterations=1)

    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # sort the contours from left-to-right and, then initialize the
    # distance colors and reference object
    (cnts, _) = contours.sort_contours(cnts)

    # loop over the contours individually
    cnt_check_list = [False for _ in range(len(cnts))]
    mod_image = copy.deepcopy(image)
    i, j = 0, 0
    # 두 object를 선택한 후, 해당 object를 감싸는 가장 작은 원 사이의 거리를 구한다.
    # 해당 거리를 distance_array에 넣는다.
    for obj1 in cnts:
        cnt_check_list[i] = True
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(obj1) < 80:  # This should be changed with relate to
            continue  # robot environment.
        # compute the rotated bounding box of the contour
        (obj1_x, obj1_y), obj1_radius = cv2.minEnclosingCircle(obj1)
        obj1_center = (int(obj1_x), int(obj1_y))
        obj1_radius = int(obj1_radius)
        cv2.circle(mod_image, obj1_center, obj1_radius, (177, 177, 177), 1)

        # TODO: getting the nearest distance between two objects
        # loop over the original points
        color = (223, 223, 223)
        orig = image.copy()
        j = 0
        for obj2 in cnts:
            if cnt_check_list[j] == True:
                j += 1
                continue
            if cv2.contourArea(obj2) < 80:
                j += 1
                continue
            (obj2_x, obj2_y), obj2_radius = cv2.minEnclosingCircle(obj2)
            obj2_center = (int(obj2_x), int(obj2_y))
            obj2_radius = int(obj2_radius)
            cv2.circle(mod_image, obj2_center, obj2_radius, (177, 177, 177), 1)

            # TODO: draw line between two intersection point
            line_tangent = (obj2_center[1]-obj1_center[1]) / (obj2_center[0]-obj1_center[0])
            inter1_plus_x = obj1_center[0] + np.sqrt(obj1_radius*obj1_radius / (1+line_tangent*line_tangent))
            inter1_plus_y = (inter1_plus_x-obj1_center[0]) * line_tangent + obj1_center[1]
            inter1_minus_x = obj1_center[0] - np.sqrt(obj1_radius*obj1_radius / (1+line_tangent*line_tangent))
            inter1_minus_y = (inter1_minus_x-obj1_center[0]) * line_tangent + obj1_center[1]

            inter2_plus_x = obj2_center[0] + np.sqrt(obj2_radius * obj2_radius / (1 + line_tangent * line_tangent))
            inter2_plus_y = (inter2_plus_x-obj2_center[0]) / line_tangent + obj2_center[1]
            inter2_minus_x = obj2_center[0] - np.sqrt(obj2_radius * obj2_radius / (1 + line_tangent * line_tangent))
            inter2_minus_y = (inter2_minus_x-obj2_center[0]) * line_tangent + obj2_center[1]

            small_x = min(obj1_center[0], obj2_center[0])
            big_x = max(obj1_center[0], obj2_center[0])
            small_y = min(obj1_center[1], obj2_center[1])
            big_y = max(obj1_center[1], obj2_center[1])
            inter1, inter2 = (0, 0), (0, 0)
            if small_x <= inter1_minus_x <= big_x:
                inter1 = (int(inter1_minus_x), int(inter1_minus_y))
            else:
                inter1 = (int(inter1_plus_x), int(inter1_plus_y))
            if small_x <= inter2_minus_x <= big_x:
                inter2 = (int(inter2_minus_x), int(inter2_minus_y))
            else:
                inter2 = (int(inter2_plus_x), int(inter2_plus_y))
            D = dist.euclidean(obj1_center, obj2_center) - obj1_radius - obj2_radius
            cv2.line(mod_image, inter1, inter2, color)
            (mid_x, mid_y) = midpoint(obj1_center, obj2_center)
            cv2.putText(mod_image, "{:.2f}".format(D), (int(mid_x), int(mid_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color,
                        2)
            cv2.imshow("distance between objects", mod_image)

            # SETTING INDEX
            # TODO: specify which object it is - and set index to those objects
            # If obj1's center's color is same as detected_obj_lists
            obj1_idx, obj2_idx = 0, 0
            for obj_num in detected_obj_lists:
                if tuple(mod_image[obj1_center[1]][obj1_center[0]]) == RL_Obj_List[obj_num][1]:
                    obj1_idx = obj_num
            for obj_num in detected_obj_lists:
                if tuple(mod_image[obj2_center[1]][obj2_center[0]]) == RL_Obj_List[obj_num][1]:
                    obj2_idx = obj_num
            distance_array[obj1_idx][obj2_idx] = D
            distance_array[obj2_idx][obj1_idx] = D
            scattering_logger.info("distance between {} and {} is {}.".format(obj1_idx, obj2_idx, D))
            cv2.waitKey(0)
            j += 1
        i += 1

    return distance_array


if __name__ == "__main__":
    image = None
    get_distance(image)
