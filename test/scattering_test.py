import argparse
from Robot_env import robot_env
# from segmentation import segmentation_graph
from object_detection import Seg_detector
from Robot_env.config import RL_Obj_List
import random
import copy

# for logging
import logging
import sys
from main import Agent
import numpy as np

import Scattering.scatter_util
import Robot_env

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--use_scatter', type=bool, default=True, help="use scattering")
parser.add_argument('--num_scattering', type=int, default=2, help="the number of scattering")
parser.add_argument('--seg_path', type=str, default="./segmentation/checkpoint/", help="segmentation checkpoint path")  # #--# 교체 예정
parser.add_argument('--detector_path', type=str, default="./object_detection/checkpoint/", help="object_detection checkpoint path")
parser.add_argument('--seg_threshold', type=float, default=0.80, help="segmentation threshold")

args = parser.parse_args()

socket_ip1 = "192.168.0.52"  # 오른쪽 팔(카메라)
socket_ip2 = "192.168.0.29"  # 왼쪽 팔

# test for scattering
# 1. Must detect neighbors
# 2. Must put neighboring objects to side
if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    segmentation_model = Seg_detector.Segment()
    robot = robot_env.Robot(socket_ip1, socket_ip2, segmentation_model, args.seg_threshold)

    # main scattering test
    obj_list = [i for i in range(9, 13)]   # 9~26번, 13, 14 제거 (커넥터)
    for target_cls in obj_list:
        logging.info("Current Target: {}".format(RL_Obj_List[target_cls][0]))
        robot.env_img_update()
        target_xyz, target_imgmean, target_pxl = robot.get_obj_pos(target_cls)
        robot.scatter(target_cls, True, target_xyz, 2, target_pxl)