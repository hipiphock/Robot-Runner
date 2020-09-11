import argparse
import logging
from Tester import Agent
import sys
import os 
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from object_detection import Seg_detector
from Robot_env import robot_env

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--use_scatter', type=bool, default=True, help="use scattering")
parser.add_argument('--num_scattering', type=int, default=2, help="the number of scattering")
parser.add_argument('--seg_path', type=str, default="./segmentation/checkpoint/", help="segmentation checkpoint path")  # #--# 교체 예정
parser.add_argument('--detector_path', type=str, default="./object_detection/checkpoint/", help="object_detection checkpoint path")
parser.add_argument('--seg_threshold', type=float, default=0.80, help="segmentation threshold")

args = parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(filename='logs/RobotTest.log', level=logging.INFO)

    segmentation_model = Seg_detector.Segment()

    socket_ip1 = "192.168.0.52"  # 오른쪽 팔(카메라)
    socket_ip2 = "192.168.0.29"  # 왼쪽 팔
    robot = robot_env.Robot(socket_ip1, socket_ip2, segmentation_model, args.seg_threshold)

    agent = Agent(robot)
    
    agent.run_object_picking_test()

    agent.run_drawer_test()

    agent.run_bin_test()

    agent.run_pen_lid_test()
    
    agent.run_penholder_test()

    agent.run_keyboard_test()