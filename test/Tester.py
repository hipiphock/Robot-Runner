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
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--use_scatter', type=bool, default=True, help="use scattering")
parser.add_argument('--num_scattering', type=int, default=2, help="the number of scattering")
parser.add_argument('--seg_path', type=str, default="./segmentation/checkpoint/", help="segmentation checkpoint path")  # #--# 교체 예정
parser.add_argument('--detector_path', type=str, default="./object_detection/checkpoint/", help="object_detection checkpoint path")
parser.add_argument('--seg_threshold', type=float, default=0.80, help="segmentation threshold")

args = parser.parse_args()

socket_ip1 = "192.168.0.52"  # 오른쪽 팔(카메라)
socket_ip2 = "192.168.0.29"  # 왼쪽 팔

logger = logging.getLogger("")

class Agent:

    def __init__(self, rob):
        self.robot = rob
        self.obj_list = [i for i in range(9, 13)]   # 9~26번, 13, 14 제거 (커넥터)
        self.obj_list += [i for i in range(21, 27)]
        self.drawer_list =  [1, 2]       # : drawer
        self.drawer_obj_list = [17, 18, 19, 20]
        self.bin_list =     [3, 4]          # : bin
        self.bin_obj_list = [15, 16]

        self.holder_list =  [5, 6]      # : 5:green     6:black
        self.pen_list =     [27, 28]    # : 27:namepen  28:marker
        self.keyboard_list= [7, 8]      # : 7:black     8:pink
        self.usb_list =     [29, 30]    # : 29:C-type   30:HDMI

        self.shuffled_list = []

    def set_obj(self, org_list):
        shuffled_list = copy.deepcopy(org_list)
        random.shuffle(shuffled_list)
        # self.shuffled_list = random.shuffle(self.obj_list)
        # self.shuffled_list = [20, 21, 22] # : 테스트용 잘되는 물체들 Usb_Big, Tape_black, Tape_white
        return shuffled_list

    def run_object_picking_test(self):
        logging.info("STARTING PICKING TEST")
        hasFind = True
        obj_list = self.set_obj(self.obj_list)
        for target_cls in obj_list:
            while 1:
                self.robot.env_img_update()
                target_xyz, target_imgmean, target_pxl = self.robot.get_obj_pos(target_cls)
                if target_xyz is None:
                    logging.warning("Can not find {}, xyz is None.".format(RL_Obj_List[target_cls][0]))
                    break
                logging.info("Current Target: {}".format(RL_Obj_List[target_cls][0]))
                distance_array = get_distance(self.robot.color_seg_img, self.robot.detected_obj_list)
                check = False
                for i in self.robot.detected_obj_list:
                    if distance_array[target_cls][i] < 10:
                        logging.info("Scattering Target: {}, {}".format(RL_Obj_List[target_cls][0], RL_Obj_List[i][0]))
                        target2_xyz, _, _ = self.robot.get_obj_pos(i)
                        self.robot.scatter_move_gripper(target_xyz, target2_xyz)
                        check = True
                        break
                if check is False:
                    break
            self.robot.grasp_placing_box(target_cls, target_imgmean, target_xyz)
    
    def run_drawer_test(self):
        logging.info("STARTING DRAWER TEST")
        drawer_xyz = None
        drawer_list = self.set_obj(self.drawer_list)
        for target_cls in drawer_list:
            if hasFind is True:
                self.robot.env_img_update()
            target_xyz, target_imgmean, target_pxl = self.robot.get_obj_pos(target_cls)
            if target_xyz is None:
                hasFind = False
                logging.warning("Can not find {}, xyz is None.".format(RL_Obj_List[target_cls][0]))
                continue

            hasFind = True
            logging.info("Current Target: {}".format(RL_Obj_List[target_cls][0]))
            drawer_xyz = target_xyz
            break
        if drawer_xyz is not None:
            obj_list = self.set_obj(self.drawer_obj_list)
            for target_cls in obj_list:
                if hasFind is True:
                   self.robot.env_img_update()
                target_xyz, target_imgmean, target_pxl = self.robot.get_obj_pos(target_cls)
                if target_xyz is None:
                    hasFind = False
                    logging.warning("Can not find {}, xyz is None.".format(RL_Obj_List[target_cls][0]))
                    continue

                hasFind = True
                logging.info("Current Target: {}".format(RL_Obj_List[target_cls][0]))
                self.robot.grasp_placing_drawer(target_cls, target_imgmean, target_xyz)
                self.robot.open_drawer(drawer_xyz)
                self.robot.grasp_place_drawer_obj(drawer_xyz)
                self.robot.close_drawer(drawer_xyz)

    def run_bin_test(self):
        logging.info("STARTING BIN TEST")
        bin_list = self.set_obj(self.bin_list)
        for bin_cls in bin_list:
            if hasFind is True:
                self.robot.env_img_update()
            bin_xyz, bin_imgmean, bin_pxl = self.robot.get_obj_pos(bin_cls)
            if bin_xyz is None:
                hasFind = False
                logging.warning("Can not find {}, xyz is None.".format(RL_Obj_List[target_cls][0]))
                continue
            hasFind = True
            obj_list = self.set_obj(self.bin_obj_list)
            for target_cls in obj_list:
                if hasFind is True:
                    self.robot.env_img_update()
                target_xyz, target_imgmean, target_pxl = rob.get_obj_pos(target_cls)
                if target_xyz is None:
                    hasFind = False
                    logging.warning("Can not find {}, xyz is None.".format(RL_Obj_List[target_cls][0]))
                    continue
                hasFind = True
                logging.info("Current Target: {}".format(RL_Obj_List[target_cls][0]))
                self.robot.grasp_placing_bin(target_cls, target_imgmean, target_xyz, bin_xyz)

    def run_penholder_test(self):
        logging.info("STARTING PENHOLDER TEST")
        holder_list = self.set_obj(self.holder_list)
        #holder_list = [9]   # : test용, 홀더와 빈, 서랍 Mask-RCNN에 포함시켜야함
        h_loc = None
        for target_cls in holder_list:
            if hasFind is True:
                self.robot.env_img_update()
            target_xyz, target_imgmean, target_pxl = self.robot.get_obj_pos(target_cls)
            if target_xyz is None:
                hasFind = False
                logging.warning("Can not find {}, xyz is None.".format(RL_Obj_List[target_cls][0]))
                continue
            hasFind = True
            logging.info("Current Target: {}".format(RL_Obj_List[target_cls][0]))
            h_loc = self.robot.grasp_holder(target_cls, target_xyz)
            break
        if h_loc is not None:
            pen_list = self.set_obj(self.pen_list)
            for target_cls in pen_list:
                if hasFind is True:
                   self.robot.env_img_update()
                target_xyz, _, target_pxl = self.robot.get_obj_pos(target_cls)
                if target_xyz is None:
                    hasFind = False
                    logging.warning("Can not find {}, xyz is None.".format(RL_Obj_List[target_cls][0]))
                    continue
                hasFind = True
                logging.info("Current Target: {}".format(RL_Obj_List[target_cls][0]))
                self.robot.grasp_pen(target_cls, target_xyz)
                self.robot.placing_toholder(h_loc)
            self.robot.holder_toplace(h_loc)

    def run_keyboard_test():
        logging.info("STARTING KEYBOARD TEST")
        key_list = self.set_obj(self.keyboard_list)
        for target_cls in key_list:
            if hasFind is True:
                self.robot.env_img_update()
            target_xyz, mean_xy, target_pxl = self.robot.get_obj_pos(target_cls)
            if target_xyz is None:
                hasFind = False
                logging.warning("Can not find {}, xyz is None.".format(RL_Obj_List[target_cls][0]))
                continue
            hasFind = True
            self.robot.grasp_placing_keyboard(target_cls, mean_xy)