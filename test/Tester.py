import sys
import os 
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Robot_env import robot_env
from Robot_env.scattering_easy import get_distance as get_distance
from object_detection import Seg_detector
from Robot_env.config import RL_Obj_List
import random
import copy

# for logging
import logging
import sys
import numpy as np

logger = logging.getLogger("Agent")

class Agent:
    def __init__(self, rob):
        self.robot = rob
        self.obj_list = [i for i in range(9, 13)]       # 9~26번, 13, 14 제거 (커넥터)
        self.obj_list += [i for i in range(21, 27)]
        self.drawer_list = [1, 2]                       # drawer
        self.drawer_obj_list = [17, 18, 19, 20]
        self.bin_list = [3, 4]                          # bin
        self.bin_obj_list = [15, 16]
        self.bottle_lid_list = [38, 39]
        self.pen_lid_list = [31, 32, 33]    
        self.holder_list = [5, 6]                       # 5:green     6:black
        self.pen_list = [27, 28]                        # 27:namepen  28:marker
        self.wide_object_list = [7, 8, 34, 35, 36, 40]  # 7:black     8:pink
        self.usb_list = [29, 30]                        # 29:C-type   30:HDMI
        self.cleaner_list = [37, 41]

        self.shuffled_list = []

    def set_obj(self, org_list):
        shuffled_list = copy.deepcopy(org_list)
        random.shuffle(shuffled_list)
        # self.shuffled_list = random.shuffle(self.obj_list)
        # self.shuffled_list = [20, 21, 22] # : 테스트용 잘되는 물체들 Usb_Big, Tape_black, Tape_white
        return shuffled_list

    def run_object_picking_test(self):
        NEEDS_UPDATE = True
        obj_list = self.set_obj(self.obj_list)
        for target_cls in obj_list:
            if NEEDS_UPDATE:
                self.robot.env_img_update()
            target_xyz, target_imgmean, target_pxl = self.robot.get_obj_pos(target_cls)
            if target_xyz is None:
                NEEDS_UPDATE = False
                logger.warning("Could not find {}, target xyz is None.".format(RL_Obj_List[target_cls][0]))
                continue
            # check whether it needs scattering or not
            NEEDS_SCATTERING = True
            while NEEDS_SCATTERING:
                distance_array = get_distance(self.robot.color_seg_img, self.robot.detected_obj_list)
                for detected_obj in self.robot.detected_obj_list:
                    # target_cls가 완전히 집을 수 있는 상태가 될 때까지 scattering을 한다.
                    if distance_array[target_cls][detected_obj] < 9:
                        logger.info("Scattering Target: {}, {}".format(RL_Obj_List[target_cls][0], RL_Obj_List[detected_obj][0]))
                        target2_xyz, _, _ = self.robot.get_obj_pos(detected_obj)
                        target_list = [target_cls, detected_obj]
                        self.robot.seg_model.emphasize_targets(self.robot.original_image, target_list)
                        self.robot.scatter_move_gripper(target_xyz, target2_xyz)
                        NEEDS_UPDATE = True
                        break
                    NEEDS_SCATTERING = False
                if NEEDS_UPDATE:
                    self.robot.env_img_update()
                    NEEDS_UPDATE = False
            if target_cls in self.picking_obj_list:
                logger.info("Current Target: {}".format(RL_Obj_List[target_cls][0]))
                self.robot.seg_model.emphasize_target(self.robot.original_image, target_cls)
                self.robot.grasp_placing_box(target_cls, target_imgmean, target_xyz)
                NEEDS_UPDATE = True
    
    def run_drawer_test(self):
        logger.info("STARTING DRAWER TEST")
        drawer_xyz = None
        drawer_list = self.set_obj(self.drawer_list)
        hasFind = True
        for target_cls in drawer_list:
            if hasFind is True:
                self.robot.env_img_update()
            target_xyz, target_imgmean, target_pxl = self.robot.get_obj_pos(target_cls)
            if target_xyz is None:
                hasFind = False
                logger.warning("Can not find {}, xyz is None.".format(RL_Obj_List[target_cls][0]))
                continue

            hasFind = True
            logger.info("Current Target: {}".format(RL_Obj_List[target_cls][0]))
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
                    logger.warning("Can not find {}, xyz is None.".format(RL_Obj_List[target_cls][0]))
                    continue

                hasFind = True
                logger.info("Current Target: {}".format(RL_Obj_List[target_cls][0]))
                self.robot.seg_model.emphasize_target(self.robot.original_image, target_cls)
                self.robot.grasp_placing_drawer(target_cls, target_imgmean, target_xyz)
                self.robot.open_drawer(drawer_xyz)
                self.robot.grasp_place_drawer_obj(drawer_xyz)
                self.robot.close_drawer(drawer_xyz)

    def run_bin_test(self):
        logger.info("STARTING BIN TEST")
        bin_list = self.set_obj(self.bin_list)
        hasFind = True
        for bin_cls in bin_list:
            if hasFind is True:
                self.robot.env_img_update()
            bin_xyz, bin_imgmean, bin_pxl = self.robot.get_obj_pos(bin_cls)
            if bin_xyz is None:
                hasFind = False
                logger.warning("Can not find {}, xyz is None.".format(RL_Obj_List[bin_cls][0]))
                continue
            hasFind = True
            obj_list = self.set_obj(self.bin_obj_list)
            for target_cls in obj_list:
                if hasFind is True:
                    self.robot.env_img_update()
                target_xyz, target_imgmean, target_pxl = self.robot.get_obj_pos(target_cls)
                if target_xyz is None:
                    hasFind = False
                    logger.warning("Can not find {}, xyz is None.".format(RL_Obj_List[target_cls][0]))
                    continue
                hasFind = True
                logger.info("Current Target: {}".format(RL_Obj_List[target_cls][0]))
                self.robot.seg_model.emphasize_target(self.robot.original_image, target_cls)
                self.robot.grasp_placing_bin(target_cls, target_imgmean, target_xyz, bin_xyz)
    
    def run_bottle_test(self):
        logging.info("STARTING BOTTLE LID TEST")
        bottle_lid_list = self.set_obj(self.bottle_lid_list)
        hasFind = True
        for target_cls in bottle_lid_list:
            if hasFind is True:
                self.robot.env_img_update()
            target_xyz, mean_xy, target_pxl = self.robot.get_obj_pos(target_cls)
            if target_xyz is None:
                hasFind = False
                logger.warning("Can not find {}, xyz is None.".format(RL_Obj_List[target_cls][0]))
                continue
            hasFind = True
            self.robot.seg_model.emphasize_target(self.robot.original_image, target_cls)
            self.robot.grasp_open_bottle_lid(target_cls, target_imgmean, target_xyz)


    def run_pen_lid_test(self):
        logging.info("STARTING PEN LID OPEN TEST")
        pen_lid_list = self.set_obj(self.pen_lid_list)
        hasFind = True
        for target_cls in pen_lid_list:
            if hasFind is True:
                self.robot.env_img_update()
            target_xyz, mean_xy, target_pxl = self.robot.get_obj_pos(target_cls)
            if target_xyz is None:
                hasFind = False
                logger.warning("Can not find {}, xyz is None.".format(RL_Obj_List[target_cls][0]))
                continue
            hasFind = True
            self.robot.seg_model.emphasize_target(self.robot.original_image, target_cls)
            self.robot.grasp_open_pen_lid(target_cls, target_imgmean, target_xyz)

    def run_penholder_test(self):
        logger.info("STARTING PENHOLDER TEST")
        holder_list = self.set_obj(self.holder_list)
        #holder_list = [9]   # : test용, 홀더와 빈, 서랍 Mask-RCNN에 포함시켜야함
        h_loc = None
        hasFind = True
        for target_cls in holder_list:
            if hasFind is True:
                self.robot.env_img_update()
            target_xyz, target_imgmean, target_pxl = self.robot.get_obj_pos(target_cls)
            if target_xyz is None:
                hasFind = False
                logger.warning("Can not find {}, xyz is None.".format(RL_Obj_List[target_cls][0]))
                continue
            hasFind = True
            logger.info("Current Target: {}".format(RL_Obj_List[target_cls][0]))
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
                    logger.warning("Can not find {}, xyz is None.".format(RL_Obj_List[target_cls][0]))
                    continue
                hasFind = True
                logger.info("Current Target: {}".format(RL_Obj_List[target_cls][0]))
                self.robot.seg_model.emphasize_target(self.robot.original_image, target_cls)
                self.robot.grasp_pen(target_cls, target_xyz)
                self.robot.placing_toholder(h_loc)
            self.robot.holder_toplace(h_loc)

    def run_keyboard_test(self):
        logger.info("STARTING KEYBOARD TEST")
        key_list = self.set_obj(self.wide_object_list)
        hasFind = True
        for target_cls in key_list:
            if hasFind is True:
                self.robot.env_img_update()
            target_xyz, mean_xy, target_pxl = self.robot.get_obj_pos(target_cls)
            if target_xyz is None:
                hasFind = False
                logger.warning("Can not find {}, xyz is None.".format(RL_Obj_List[target_cls][0]))
                continue
            hasFind = True
            self.robot.seg_model.emphasize_target(self.robot.original_image, target_cls)
            self.robot.grasp_placing_keyboard(target_cls, mean_xy)