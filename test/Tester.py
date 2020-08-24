import argparse
import logging
from builtins import print

from Robot_env import robot_env
# from segmentation import segmentation_graph
from object_detection import Seg_detector
from Robot_env.config import RL_Obj_List
import random
import copy
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--use_scatter', type=bool, default=True, help="use scattering")
parser.add_argument('--num_scattering', type=int, default=2, help="the number of scattering")
parser.add_argument('--seg_path', type=str, default="./segmentation/checkpoint/", help="segmentation checkpoint path")  # #--# 교체 예정
parser.add_argument('--detector_path', type=str, default="./object_detection/checkpoint/", help="object_detection checkpoint path")
parser.add_argument('--seg_threshold', type=float, default=0.60, help="segmentation threshold")

args = parser.parse_args()

camera_robot_ip  = "192.168.0.52"  # 오른쪽 팔(카메라)
gripper_robot_ip = "192.168.0.29"  # 왼쪽 팔


class Agent:
    """
    Main class that handles total procedure of test.
    It has three test member functions.
    """
    def __init__(self, rob):
        self.robot = rob
        self.obj_list = [i for i in range(9, 27)]   # 9~26번, 13, 14 제거 (커넥터)
        self.obj_list.remove(13)        # : small connector
        self.obj_list.remove(14)        # : big connector

        obj_list_add=[i for i in range(31,42)]
        self.obj_list+=obj_list_add

        self.holder_list    = [5, 6]                # 5:green     6:black
        self.pen_list       = [27, 28, 31, 32, 33]  # 27:namepen  31:silver_namepen
                                                    # 28:marker   32:black_marker
                                                    # 32:blue_marker
        self.usb_list       = [29, 30]              # 29:C-type   30:HDMI
        self.keyboard_list  = [7, 8]                # 7:black     8:pink
        self.board_list     = [34, 35]              # 34:black    35:pink
        self.bottle_list    = [38, 39]              # 38:apricot  39:grey
        self.book_list      = [36, 40]              # 36:purple   40:white
        self.shuffled_list = []

    def set_obj(self, org_list):
        shuffled_list = copy.deepcopy(org_list)
        random.shuffle(shuffled_list)
        # self.shuffled_list = random.shuffle(self.obj_list)
        # self.shuffled_list = [20, 21, 22] # : 테스트용 잘되는 물체들 Usb_Big, Tape_black, Tape_white
        return shuffled_list

    def run_object_picking_test(self):
        """
        Run object picking test.
        This test pick up objects and put them in basket.
        """
        rob = self.robot
        episode_num = 1
        rob.rob1.getl()

        obj_list = self.set_obj(self.obj_list)
        for target_obj in obj_list:

            rob.env_img_update()    # update image

            target_xyz, target_imgmean, target_pxl = rob.get_obj_pos(target_obj)
            if target_xyz is None:
                print("!!>>sys : Can't find {} xyz is None ".format(RL_Obj_List[target_obj][0]))
                logging.info("Could not find {}, xyz is None".format(RL_Obj_List[target_obj][0]))
                continue
            logging.info("Current Target: {}".format(RL_Obj_List[target_obj][0]))
            rob.grasp_placing_box(target_obj, target_imgmean, target_xyz)

    def run_penholder_test(self):

        rob = self.robot
        episode_num = 1
        rob.rob1.getl()
        # rob.rob1.movel([0.5129180147348896, -0.15589696029984967, -0.11966152182033941+0.10, 2.110665698267417, -2.2796431536929087, 0.010529626934130714], 1.0, 1.0)
        # rob.rob1.movel([0.5129180147348896, -0.15589696029984967, -0.11966152182033941, 2.110665698267417, -2.2796431536929087, 0.010529626934130714], 1.0, 1.0)

        # ---- ---- ---- ---- Pen ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
        holder_list = self.set_obj(self.holder_list)
        h_loc = None
        for target_cls in holder_list:
            rob.env_img_update()
            target_xyz, target_imgmean, target_pxl = rob.get_obj_pos(target_cls)
            if target_xyz is None:
                logging.info("Could not find {}, xyz is None.")
                continue
        
            logging.debug("Current Target: {}".format(RL_Obj_List[target_cls][0]))
            h_loc = rob.grasp_holder(target_cls, target_xyz)
            break
        
        pen_list = self.set_obj(self.pen_list)
        for target_cls in pen_list:
        
            rob.env_img_update()
        
            target_xyz, _, target_pxl = rob.get_obj_pos(target_cls)
            if target_xyz is None:
                logging.info("Could not find {}, xyz is None.")
                continue
        
            logging.info("Current Target: {}".format(RL_Obj_List[target_cls][0]))
            rob.grasp_pen(target_cls, target_xyz)
        
            # : 이미지 찍을지 말지 결정 필요
            # : 안찍을시 하드코딩
            rob.placing_toholder(h_loc)
            
        rob.holder_toplace(h_loc)

    def run_keyboard_test(self):

        rob = self.robot
        episode_num = 1
        rob.rob1.getl()
        # rob.rob1.movel([0.5129180147348896, -0.15589696029984967, -0.11966152182033941+0.10, 2.110665698267417, -2.2796431536929087, 0.010529626934130714], 1.0, 1.0)
        # rob.rob1.movel([0.5129180147348896, -0.15589696029984967, -0.11966152182033941, 2.110665698267417, -2.2796431536929087, 0.010529626934130714], 1.0, 1.0)
        while True:

            print("------------- Test No.{} ------------".format(episode_num))
            rob.reset()

            # ---- ---- ---- ---- Keyboard ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
            key_list = self.set_obj(self.keyboard_list)
            for target_cls in key_list:

                rob.env_img_update()

                target_xyz, mean_xy, target_pxl = rob.get_obj_pos(target_cls)
                if target_xyz is None:
                    print("!!>>sys : Can't find {} xyz is None ".format(RL_Obj_List[target_cls][0]))
                    continue

                rob.grasp_placing_keyboard(target_cls, mean_xy)

if __name__ == "__main__":
    logging.basicConfig(filename='logs/RobotTest.log', level=logging.INFO)

    segmentation_model = Seg_detector.Segment()
    robot = robot_env.Robot(camera_robot_ip, gripper_robot_ip, segmentation_model, args.seg_threshold)

    agent = Agent(robot)
    agent.run_object_picking_test()