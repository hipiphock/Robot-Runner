from scipy.spatial import distance as dist
import imutils
from imutils import contours
import numpy as np
import cv2
import copy
import logging
from Robot_env.config import RL_Obj_List

"""
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


def get_distance(image, detected_obj_lists):
    """
    Get the distance between objects
    :param image: color segment image from env_img_update()
    :param detected_obj_lists: also from env_img_update()
    :return:
    """
    obj_to_be_ignored = list([i for i in range(8)])

    # distance_array is an array that represents distance between two objects.
    # For example, if cup_red(9) and big_box(12)'s distance is 11.82, then distance_aray[9][12] = 11.82.
    distance_array = np.ones((50, 50))
    distance_array = np.full_like(distance_array, 999.99)
    # num_objs = len(detected_obj_lists)
    # midpoint_list = list(num_objs)
    # for i in midpoint_list:
    #     i.append([])

    # load the image, convert it to grayscale, and blur it slightly
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # gray = cv2.GaussianBlur(gray, (7, 7), 0)
    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged = cv2.Canny(rgb_img, 50, 100)
    # edged = cv2.dilate(edged, None, iterations=1)
    # edged = cv2.erode(edged, None, iterations=1)

    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # sort the contours from left-to-right and, then initialize the
    # distance colors and reference object
    (cnts, _) = contours.sort_contours(cnts)

    # loop over the contours individually
    mod_image = copy.deepcopy(image)
    # 두 object를 선택한 후, 해당 object를 감싸는 가장 작은 원 사이의 거리를 구한다.
    # 해당 거리를 distance_array에 넣는다.
    for obj1 in cnts:
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(obj1) < 20:  # This should be changed with relate to
            continue                    # robot environment.
        # compute the rotated bounding box of the contour
        (obj1_x, obj1_y), obj1_radius = cv2.minEnclosingCircle(obj1)
        obj1_center = (int(obj1_x), int(obj1_y))
        obj1_radius = int(obj1_radius)
        ignore = False
        for obj_num in obj_to_be_ignored:
            if tuple(image[obj1_center[1]][obj1_center[0]]) == RL_Obj_List[obj_num][1]:
                scattering_logger.debug("object {} must be ignored.".format(obj_num))
                ignore = True
                break
        if ignore:
            continue
        cv2.circle(mod_image, obj1_center, obj1_radius, (253, 253, 253), 1)

        # loop over the original points
        color = (253, 253, 253)
        for obj2 in cnts:
            if cv2.contourArea(obj2) < 20:
                continue
            (obj2_x, obj2_y), obj2_radius = cv2.minEnclosingCircle(obj2)
            obj2_center = (int(obj2_x), int(obj2_y))
            obj2_radius = int(obj2_radius)
            # ignoring same object and ignore_obj
            ignore = False
            for obj_num in obj_to_be_ignored:
                if tuple(image[obj2_center[1]][obj2_center[0]]) == RL_Obj_List[obj_num][1]:
                    scattering_logger.debug("object {} must be ignored.".format(obj_num))
                    ignore = True
                    break
                elif tuple(image[obj2_center[1]][obj2_center[0]]) == tuple(image[obj1_center[1]][obj1_center[0]]):
                    ignore = True
            if ignore:
                continue
            cv2.circle(mod_image, obj2_center, obj2_radius, color, 1)

            if (obj2_center[0]-obj1_center[0]) == 0:
                # Case: both object's x pos is same
                if obj1_center[1] > obj2_center[1]:
                    inter1_y = obj1_center[1] - obj1_radius
                    inter2_y = obj2_center[1] + obj2_radius
                else:
                    inter1_y = obj1_center[1] + obj1_radius
                    inter2_y = obj2_center[1] - obj2_radius
                inter1 = (obj1_center[0], inter1_y)
                inter2 = (obj2_center[0], inter2_y)

            else:
                line_tangent = (obj2_center[1]-obj1_center[1]) / (obj2_center[0]-obj1_center[0])

                inter1_plus_x = obj1_center[0] + np.sqrt(obj1_radius*obj1_radius / (1+line_tangent*line_tangent))
                inter1_plus_y = (inter1_plus_x-obj1_center[0]) * line_tangent + obj1_center[1]
                inter1_minus_x = obj1_center[0] - np.sqrt(obj1_radius*obj1_radius / (1+line_tangent*line_tangent))
                inter1_minus_y = (inter1_minus_x-obj1_center[0]) * line_tangent + obj1_center[1]

                inter2_plus_x = obj2_center[0] + np.sqrt(obj2_radius * obj2_radius / (1 + line_tangent * line_tangent))
                inter2_plus_y = (inter2_plus_x-obj2_center[0]) * line_tangent + obj2_center[1]
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
            cv2.putText(mod_image, "{:.2f}".format(D), (int(mid_x), int(mid_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.imshow("Distance between objects", mod_image)

            # SETTING INDEX
            # If obj1's center's color is same as detected_obj_lists
            obj1_idx, obj2_idx = 0, 0
            obj_to_be_ignored = list([i for i in range(8)])
            ignore = False
            for obj_num in detected_obj_lists:
                if tuple(image[obj1_center[1]][obj1_center[0]]) == RL_Obj_List[obj_num][1]:
                    if obj_num in obj_to_be_ignored:
                        ignore = True
                        break
                    obj1_idx = obj_num
                    break
            for obj_num in detected_obj_lists:
                if tuple(image[obj2_center[1]][obj2_center[0]]) == RL_Obj_List[obj_num][1]:
                    if obj_num in obj_to_be_ignored:
                        ignore = True
                        break
                    elif obj_num == obj1_idx:
                        ignore = True
                        break
                    obj2_idx = obj_num
                    break
            if ignore:
                continue
            distance_array[obj1_idx][obj2_idx] = D
            distance_array[obj2_idx][obj1_idx] = D
            scattering_logger.info("distance between {} and {} is {}.".format(RL_Obj_List[obj1_idx][0], RL_Obj_List[obj2_idx][0], D))
            cv2.waitKey(1)
            # cv2.imwrite("object_distance.png", mod_image)
    return distance_array


if __name__ == "__main__":
    image = None
