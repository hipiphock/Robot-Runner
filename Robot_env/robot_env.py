# Robot
import urx
import logging
from matplotlib.mlab import prctile
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
from Realsense_env.realsenseForTray_RL import *
import socket

# Solve FK, IK
import cmath
import math
from math import cos as cos
from math import sin as sin
from math import atan2 as atan2
from math import acos as acos
from math import asin as asin
from math import sqrt as sqrt
from math import pi as pi
import math3d as m3d

# utils
import random
import logging
from Robot_env.config import RL_Obj_List
from Robot_env.scattering_path import *
#
from Robot_env import socket_communication
from Robot_env import robot_util
from Robot_env.robot_util import *

import time
import sys
import numpy as np
import copy
import logging

logger = logging.getLogger("Robot")
# handler = logging.StreamHandler(sys.stderr)
# handler.setLevel(logging.WARNING)
# logger.addHandler(handler)


class Robot:
    def __init__(self, socket_ip1, socket_ip2, segmentation_model=None, threshold=0.60):
        self.rob1 = robot_util.Robot_util(socket_ip1)
        time.sleep(1)
        self.rob2 = robot_util.Robot_util(socket_ip2)
        time.sleep(1)

        self.gripper1 = self.rob1.gripper
        self.gripper2 = self.rob2.gripper
        self.acc = 1
        self.vel = 1

        # Camera
        self.global_cam = Realsense()
        self.env_img = None

        # Segmentation Model
        self.seg_model = segmentation_model
        self.seg_threshold = threshold
        self.seg_img = None
        self.color_seg_img = None
        self.is_singular = False
        self.env_show = None  # :20200102

        # Set Position
        self.home = np.deg2rad([0.0, -90.0, 0.0, -90.0, 0.0, 0.0])
        self.initial_pose1 = np.deg2rad([-20.0, -110.0, -70.0, -90.0, 90.0, -20.0])
        self.initial_pose2 = np.deg2rad([20.0, -70.0, 70.0, -90.0, -90.0, 20.0])
        self.cam_position = np.deg2rad([0.3209, -113.0970, -4.5383, -152.3580, 89.6613, 1.2152])  # : 교체 후

        # Robot Dynamics & Kinematics Parameters
        self.ur5_a = [0, -0.425, -0.39225, 0, 0, 0]
        self.ur5_d = [0.089159, 0, 0, 0.10915, 0.09465, 0.0823]
        self.alp = [np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0]

        self.goal = [0, 0, 0, 0, 0, 0]

        self.img_num = 0

        # Place Type
        self.drawer_pos = None
        self.drawer_pos_open = None
        self.bin_pos = None
        self.drawer_type = 0
        self.bin_type = 0

        # object position
        self.target_cls = 0
        self.obj_pos = None  # (x, y, z)
        self.eigen_value = np.zeros([2])
        self.color_path = None
        self.threshold = 3.2
        self.obj_type = 0  # Obj_type, 0 = drawer, 1 = bin, 2 = box
        self.start_pt = [0, 0]
        self.end_pt = [0, 0]

        self.detected_obj_list = []

        # Variables
        self.default_tcp = [0, 0, 0.153, 0, 0, 0]  # (x, y, z, rx, ry, rz)
        self.done = False

        # Reset Environment
        self.rob1.set_tcp(self.default_tcp)
        self.rob2.set_tcp(self.default_tcp)
        self.use_scatter = True

        self.x_boundary = [-0.810, -0.245]  # : 72번 기준 ㅡ
        self.y_boundary = [-0.305, 0.515]  # : 72번 기준 l
        self.z_lift = 0.015
        self.z_tray = -0.105

        self.robot_dual_control(rob_total_pose=self.home, vel=1.0, acc=1.0)
        self.action_gripper_both_open()

        logger.warning("ROBOT ENVIRONMENT READY.")

        # self.robot_pose_chceck()

    def robot_dual_control(
            self, rob_total_pose=None, vel=0.1, acc=0.1,
            rob1_pose=None, rob1_vel=0.1, rob1_acc=0.1,
            rob2_pose=None, rob2_vel=0.1, rob2_acc=0.1, is_r1_l=False, is_r2_l=False):

        if is_r1_l is False:
            r1_joints = True
        else:
            r1_joints = False

        if is_r2_l is False:
            r2_joints = True
        else:
            r2_joints = False

        if rob_total_pose is not None and is_r1_l is False and is_r2_l is False:
            self.rob1.movej(rob_total_pose, vel, acc, False)
            self.rob2.movej(rob_total_pose, vel, acc, False)

            self.rob1._wait_for_move(rob_total_pose, joints=r1_joints)
            self.rob2._wait_for_move(rob_total_pose, joints=r2_joints)

        elif rob_total_pose is not None and is_r1_l is True and is_r2_l is True:
            self.rob1.movel(rob_total_pose, vel, acc, False)
            self.rob2.movel(rob_total_pose, vel, acc, False)

            self.rob1._wait_for_move(rob_total_pose, joints=r1_joints)
            self.rob2._wait_for_move(rob_total_pose, joints=r2_joints)

        else:
            if rob1_pose is not None:
                if is_r1_l is False:
                    self.rob1.movej(rob1_pose, rob1_vel, rob1_acc, False)
                else:
                    self.rob1.movel(rob1_pose, rob1_vel, rob1_acc, False)
            if rob2_pose is not None:
                if is_r2_l is False:
                    self.rob2.movej(rob2_pose, rob2_vel, rob2_acc, False)
                else:
                    self.rob2.movel(rob2_pose, rob2_vel, rob2_acc, False)

            self.rob1._wait_for_move(rob1_pose, joints=r1_joints)
            self.rob2._wait_for_move(rob2_pose, joints=r2_joints)

    def reset(self):
        self.robot_dual_control(rob_total_pose=self.home, vel=1.0, acc=1.0)
        self.action_gripper_both_open()

        self.env_img = None
        self.seg_img = None
        self.color_seg_img = None
        self.img_num = 0

        self.is_singular = False

        self.goal = [0, 0, 0, 0, 0, 0]  # : robot loc
        self.done = False

        self.drawer_pos = None
        self.drawer_pos_open = None
        self.bin_pos = None
        self.drawer_type = 0
        self.bin_type = 0

        self.target_cls = 0
        self.obj_pos = None  # (x, y, z)
        self.eigen_value = np.zeros([2])
        self.color_path = None
        self.threshold = 3.2
        self.obj_type = 0  # Obj_type, 0 = drawer, 1 = bin, 2 = box
        self.start_pt = [0, 0]
        self.end_pt = [0, 0]

        self.use_scatter = True

    def env_img_update(self):
        self.rob1.movej(self.cam_position, 1.0, 1.0)
        self.env_img = self.global_cam.capture()
        self.seg_img, self.color_seg_img, _ = self.seg_model.total_run(self.env_img, self.seg_threshold)

        env_img_shape = self.env_img.shape
        colorseg_shape = self.color_seg_img.shape
        self.env_show = cv2.resize(self.env_img, (int(env_img_shape[1] / 2), int(env_img_shape[0] / 2)))
        colorseg_show = cv2.resize(self.color_seg_img, (int(colorseg_shape[1] / 2), int(colorseg_shape[0] / 2)))

        cv2.namedWindow("env_show")
        cv2.namedWindow("colorseg_show")

        cv2.imshow("env_show", self.env_show)
        cv2.moveWindow("env_show", 0, 0)
        cv2.imshow("colorseg_show", colorseg_show)
        cv2.moveWindow("colorseg_show", 0, 390)
        cv2.waitKey(1)

        cv2.imshow("env_show", self.env_show)
        cv2.moveWindow("env_show", 0, 0)
        cv2.imshow("colorseg_show", colorseg_show)
        cv2.moveWindow("colorseg_show", 0, 390)
        cv2.waitKey(1)

    # = rob2 Calibration to rob1 calibration
    def robot_action_mod(self, xyz):  # : 최소2.5mm 최대5mm정도의 오차발생 (크리티컬하지 않음)
        xyz_mod = xyz.copy()
        xyz_mod[0] = -xyz[0]
        xyz_mod[1] = -xyz[1] + 0.185
        return xyz_mod

    def get_obj_pos(self, target_cls, use_imgpoint=None):
        pxl_list = None
        mean_xy = None
        try:
            if use_imgpoint is True:  # : 이미지상의 임의의 포인트만 계산
                pxl_list = target_cls
                mean_xy = target_cls
            else:
                pxl_list = np.argwhere(self.seg_img == target_cls)
                mean_xy = np.copy(np.mean(pxl_list, 0))
        except np.linalg.LinAlgError:
            logger.warning("np.linalg.LinAlgError")
            return None, None, None

        xyz = None
        if pxl_list is not None:
            xyz = self.global_cam.pxl2xyz(pxl_list, mean_xy)

        self.obj_pos = xyz
        return [xyz, mean_xy, pxl_list]

    def angle_detect(self, target_cls):
        obj_angle = 1.1
        w = 1.1
        h = 1.1
        temp_ = self.get_boxed_angle(target_cls)
        if temp_ is not None:
            obj_angle, w, h = temp_
        if obj_angle is None or obj_angle == 1.1:
            obj_angle = 0
            logger.info("target{} : angle is not detected".format(target_cls))
        if w is None or w == 1.1:
            w = 0
            logger.info("target{} : w is not detected".format(target_cls))
        if h is None or h == 1.1:
            h = 0
            logger.info("target{} : h is not detected".format(target_cls))
        return obj_angle, w, h

    def action_gripper_both(self, value):
        script1 = self.gripper1._get_new_urscript()
        script2 = self.gripper2._get_new_urscript()

        sleep = 1.0
        script1._set_gripper_position(value)
        script1._sleep(sleep)
        script2._set_gripper_position(value)
        script2._sleep(sleep)

        self.rob1.send_program(script1())
        self.rob2.send_program(script2())

        time.sleep(sleep)

    def action_gripper_both_open(self):
        script1 = self.gripper1._get_new_urscript()
        script2 = self.gripper2._get_new_urscript()

        sleep = 1.0
        script1._set_gripper_position(0)
        script1._sleep(sleep)
        script2._set_gripper_position(0)
        script2._sleep(sleep)

        self.rob1.send_program(script1())
        self.rob2.send_program(script2())

        time.sleep(sleep)

    def action_gripper_both_close(self):
        script1 = self.gripper1._get_new_urscript()
        script2 = self.gripper2._get_new_urscript()

        sleep = 1.0
        script1._set_gripper_position(255)
        script1._sleep(sleep)
        script2._set_gripper_position(255)
        script2._sleep(sleep)

        self.rob1.send_program(script1())
        self.rob2.send_program(script2())

        time.sleep(sleep)

    def solve_FK(self, th):
        T = np.eye(4)
        return_T = np.eye(4)

        for i in range(6):
            T[0, 0] = cos(th[i])
            T[0, 1] = -sin(th[i]) * cos(self.alp[i])
            T[0, 2] = sin(th[i]) * sin(self.alp[i])
            T[0, 3] = self.ur5_a[i] * cos(th[i])

            T[1, 0] = sin(th[i])
            T[1, 1] = cos(th[i]) * cos(self.alp[i])
            T[1, 2] = -cos(th[i]) * sin(self.alp[i])
            T[1, 3] = self.ur5_a[i] * sin(th[i])

            T[2, 0] = 0
            T[2, 1] = sin(self.alp[i])
            T[2, 2] = cos(self.alp[i])
            T[2, 3] = self.ur5_d[i]

            T[3, 0] = 0
            T[3, 1] = 0
            T[3, 2] = 0
            T[3, 3] = 1

            return_T = (return_T @ T)

        pose_vector = m3d.Transform(return_T).pose_vector
        return pose_vector

    def robot_pose_chceck(self):
        self.default_tcp = [0, 0, 0.153, 0, 0, 0]  # (x, y, z, rx, ry, rz)
        self.rob1.set_tcp(self.default_tcp)
        self.rob2.set_tcp(self.default_tcp)
        self.home = np.deg2rad([0.0, -90.0, 0.0, -90.0, 0.0, 0.0])
        self.robot_dual_control(rob_total_pose=self.home, vel=1.0, acc=1.0)
        self.action_gripper_both_close()

        self.rob2.movej(np.deg2rad([0.0, -90.0, 30.0, -30.0, -90.0, 0.0]), 0.5, 0.5)  # : 디버깅용
        self.rob2.movel([-0.292500, -0.10883, 0.618, 2.2215, 2.2215, 0], 0.5, 0.5)  # : 디버깅용
        self.action_gripper_both_close()

        self.rob2.movel([-0.779220, -0.141110, -0.092, 2.2215, 2.2215, 0], 0.5, 0.5)  # : 디버깅용
        l_u2 = self.rob2.getl()
        self.rob2.movel([-0.776810, 0.354620, -0.092, 2.2215, 2.2215, 0], 0.5, 0.5)  # : 디버깅용
        r_u2 = self.rob2.getl()
        self.rob2.movel([-0.276740, 0.353080, -0.092, 2.2215, 2.2215, 0], 0.5, 0.5)  # : 디버깅용
        r_d2 = self.rob2.getl()
        self.rob2.movel([-0.279445, -0.144245, -0.092, 2.2215, 2.2215, 0], 0.5, 0.5)  # : 디버깅용
        l_d2 = self.rob2.getl()

        self.robot_dual_control(rob_total_pose=self.home, vel=1.0, acc=1.0)
        self.action_gripper_both_close()

        #
        self.rob1.movej(np.deg2rad([0.0, -90.0, -30.0, -150.0, 90.0, 0.0]), 0.5, 0.5)  # : 디버깅용
        self.rob1.movel([0.29158, -0.10883, 0.618, 2.2215, -2.2215, 0], 0.5, 0.5)  # : 디버깅용
        self.action_gripper_both_close()

        self.rob1.movel([0.780640, 0.326500, -0.0960, 2.2215, -2.2215, 0], 0.5, 0.5)  # : 디버깅용
        l_u1 = self.rob1.getl()
        self.rob1.movel([0.777835, -0.168790, -0.0960, 2.2215, -2.2215, 0], 0.5, 0.5)  # : 디버깅용
        r_u1 = self.rob1.getl()
        self.rob1.movel([0.278850, -0.168200, -0.0960, 2.2215, -2.2215, 0], 0.5, 0.5)  # : 디버깅용
        r_d1 = self.rob1.getl()
        self.rob1.movel([0.279690, 0.328925, -0.0960, 2.2215, -2.2215, 0], 0.5, 0.5)  # : 디버깅용
        l_d1 = self.rob1.getl()

        self.robot_dual_control(rob_total_pose=self.home, vel=1.0, acc=1.0)
        self.action_gripper_both_close()

        ######
        l_u1_ = copy.deepcopy(l_u1)
        l_u1_[1] = -l_u1_[1] + 0.185

        r_u1_ = copy.deepcopy(r_u1)
        r_u1_[1] = -r_u1_[1] + 0.185

        r_d1_ = copy.deepcopy(r_d1)
        r_d1_[1] = -r_d1_[1] + 0.185

        l_d1_ = copy.deepcopy(l_d1)
        l_d1_[1] = -l_d1_[1] + 0.185
        ####################################################################
        # : 실험 결과 # UR 반복 정밀도 +/- 0.1mm -> 실험시 평균 2.4~2.5mm 최소0.4~0.5mm 최대 5mm

        self.robot_dual_control(rob_total_pose=self.home, vel=1.0, acc=1.0)
        self.action_gripper_both_open()

    # = 각도 검출
    def get_boxed_angle(self, target_cls):
        cls_points = np.argwhere(self.seg_img == target_cls)  # 새로운 어레이가 타겟 클래스랑 같은 좌표를 찾음
        empty_img = np.zeros((720, 1280), dtype=np.uint8)

        for [y, x] in cls_points:  # 타겟클레스와 같은 점들에 대해
            empty_img[y, x] = 64  # 표시전용 (아무숫자 가능)

        time_str = time.strftime('%Y%m%d-%H-%M-%S', time.localtime(time.time()))
        cv2.imwrite("./_/test_img/{}__{}_empty_img".format(time_str, target_cls) + ".png", empty_img)

        ret, thr = cv2.threshold(empty_img, 1, 127, 0)  # 스레스 홀딩했을시 출력할 값
        cv2.imwrite("./_/test_img/{}__{}_thresholding(empty_img)".format(time_str, target_cls) + ".png", thr)

        con_img, contour, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # ?? 경계선(등고선) 그리기?
        for cnt in contour:
            rect = cv2.minAreaRect(cnt)
            angle = rect[2]
            w, h = rect[1]

            box_cont = np.int0(cv2.boxPoints(rect))
            box_img = cv2.drawContours(thr, [box_cont], 0, 255, 1)
            cv2.imwrite("./_/test_img/{}__{}_drawContours(box_img)".format(time_str, target_cls) + ".png", box_img)

            if w > h:  # Long axes
                return angle - 90, h, w
            else:  # Short axes
                return angle, w, h

    def scatter_move_gripper(self, start, end):
        """
        Easy. Move gripper toward the middle of two objects
        """
        # pre-computation
        ortho_mid = (start + end) / 2
        distance = dist.euclidean(start[:2], end[:2]) / 2
        ortho_start, ortho_end = [], [] # 기울기와 삼각함수로 좌표 알아내기
        # helper
        ortho_vector = -((end[0]-start[0])/(end[1]-start[1]))
        # ortho_line_y = lambda x : ortho_vector*(x-ortho_mid[0])+ortho_mid[1]
        ortho_angle = np.arctan(ortho_vector)
        ortho_start = ortho_mid + list(3*distance*x for x in [np.cos(ortho_angle), np.sin(ortho_angle), 0])
        ortho_end = ortho_mid - list(3*distance*x for x in [np.cos(ortho_angle), np.sin(ortho_angle), 0])

        logger.info("Starting Scattering")
        back_pose = np.deg2rad([0.0, 0.0, -90.0, -90.0, 0.0, 0.0])
        starting_pose = np.deg2rad([90.0, -100.0, 120.0, -110.0, -90.0, 0])
        placing_pose = np.deg2rad([90.0, -120.0, 140.0, -110.0, -90.0, 0])
        self.robot_dual_control(rob1_pose=back_pose, rob1_vel=1.0, rob1_acc=1.0,
                                rob2_pose=starting_pose, rob2_vel=0.75, rob2_acc=0.75)

        # 로봇의 현재 위치 기록
        rob2_loc = self.rob2.getl()
        rob2_prev_loc = copy.deepcopy(rob2_loc)
        # 로봇의 x좌표먼저 이동
        rob2_loc[0] = ortho_start[0]
        self.rob2.movel(rob2_loc, 0.5, 0.5)
        # move to ortho_start
        rob2_loc[1] = ortho_start[1]
        self.rob2.movel(rob2_loc, 0.5, 0.5)
        # handle gripper angle
        robot_joint = self.rob2.getj()
        if ortho_angle < -120:
            robot_joint[5] = np.deg2rad(ortho_angle + np.rad2deg(robot_joint[0]) + 90.0)
        elif 120 < ortho_angle:
            robot_joint[5] = np.deg2rad(ortho_angle + np.rad2deg(robot_joint[0]) - 90.0)
        else:
            robot_joint[5] = np.deg2rad(ortho_angle + np.rad2deg(robot_joint[0]))
        self.rob2.movej(robot_joint, acc=1.0, vel=1.0)

        self.gripper2.close_gripper()
        rob2_loc = self.rob2.getl()
        rob2_loc[2] = ortho_mid[2]
        self.rob2.movel(rob2_loc)
        # move to ortho_mid
        rob2_loc[0] = ortho_mid[0]
        rob2_loc[1] = ortho_mid[1]
        self.rob2.movel(rob2_loc, 0.5, 0.5)
        self.gripper2.value_gripper(122)
        # move to ortho_end
        rob2_loc[0] = ortho_end[0]
        rob2_loc[1] = ortho_end[1]
        self.rob2.movel(rob2_loc, 0.5, 0.5)
        # FINISHED, go home
        rob2_loc[0] = ortho_start[0]
        rob2_loc[1] = ortho_start[1]
        self.rob2.movel(rob2_loc, 0.4, 0.4)
        rob2_loc[2] = rob2_prev_loc[2]
        self.rob2.movel(rob2_loc, 0.5, 0.5)
        rob2_loc[1] = rob2_prev_loc[1]
        self.rob2.movel(rob2_loc, 0.5, 0.5)
        rob2_loc[0] = rob2_prev_loc[0]
        self.rob2.movel(rob2_loc, 0.5, 0.5)

    # ---- ---- ---- ---- Picking ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
    def grasp_placing_box(self, target_cls, target_imgmean, obj_pos=None, ):
        if obj_pos is None:
            logger.warning("target_position is None.")
            return
        else:
            target_pos = copy.deepcopy(obj_pos)  # --
            logger.info("target_pos: {}".format(target_pos))

            if (self.x_boundary[0] < target_pos[0] < self.x_boundary[1]) and (
                    self.y_boundary[0] < target_pos[1] < self.y_boundary[1]):

                back_pose = np.deg2rad([0.0, 0.0, -90.0, -90.0, 0.0, 0.0])
                starting_pose = np.deg2rad([90.0, -100.0, 120.0, -110.0, -90.0, 0])
                placing_pose = np.deg2rad([90.0, -120.0, 140.0, -110.0, -90.0, 0])
                self.robot_dual_control(rob1_pose=back_pose, rob1_vel=1.0, rob1_acc=1.0,
                                        rob2_pose=starting_pose, rob2_vel=0.75, rob2_acc=0.75)
                # 로봇의 현재 위치 기록
                rob2_loc = self.rob2.getl()
                rob2_loc[0] = rob2_loc[0] - 0.0015
                rob2_preloc = copy.deepcopy(rob2_loc)
                # 로봇의 x좌표먼저 이동
                rob2_preloc[0] = target_pos[0]
                self.rob2.movel(rob2_preloc, 0.5, 0.5)
                # 타겟 좌표로 이동
                rob2_loc[:2] = target_pos[:2]
                self.rob2.movel(rob2_loc, 0.5, 0.5)

                robot_joint = self.rob2.getj()
                obj_angle, w, h = self.angle_detect(target_cls)
                temp_angle = copy.deepcopy(robot_joint[5])
                gripper_angle = obj_angle + np.rad2deg(robot_joint[0])
                # handle gripper angle
                if obj_angle < -120:
                    robot_joint[5] = np.deg2rad(obj_angle + np.rad2deg(robot_joint[0]) + 180.0)
                elif 120 < obj_angle:
                    robot_joint[5] = np.deg2rad(obj_angle + np.rad2deg(robot_joint[0]) - 180.0)
                else:
                    robot_joint[5] = np.deg2rad(obj_angle + np.rad2deg(robot_joint[0]))
                self.rob2.movej(robot_joint, acc=1.0, vel=1.0)

                env_show = self.env_show.copy()
                env_show = cv2.ellipse(env_show, (int(target_imgmean[1] / 2), int(target_imgmean[0] / 2)), (0, 10),
                                       obj_angle, 0, 360, (0, 128, 0))
                env_show = cv2.ellipse(env_show, (int(target_imgmean[1] / 2), int(target_imgmean[0] / 2)), (40, 0),
                                       obj_angle, 0, 360, (0, 255, 0))
                cv2.imshow("env_show", env_show)
                cv2.moveWindow("env_show", 0, 0)
                cv2.waitKey(2)

                robot_loc_after = self.rob2.getl()
                robot_loc_after[2] = copy.deepcopy(robot_loc_after[2]) - 0.246
                self.rob2.movel(robot_loc_after, 0.5, 0.5)

                self.gripper2.close_gripper()

                self.rob2.movel(rob2_loc, 0.5, 0.5)
                self.rob2.movel(rob2_preloc, 0.5, 0.5)
                self.rob2.movej(starting_pose, 1.5, 1.5)
                self.rob2.movej(placing_pose, 1.5, 1.5)
                self.gripper2.open_gripper()
                self.rob2.movej(starting_pose, 1.5, 1.5)

            else:
                logger.warning("%s is out of Safe Boundary" % RL_Obj_List[self.target_cls][0])
                self.obj_pos = None


    def grasp_placing_bin(self, target_cls, target_imgmean, obj_pos=None, bin_pos=None):
        if obj_pos is None:
            logger.warning("target_pos is None")
            return
        else:
            target_pos = copy.deepcopy(obj_pos)  # --
            logger.info("target_pos : {}".format(target_pos))
            if (self.x_boundary[0] < target_pos[0] < self.x_boundary[1]) and (
                    self.y_boundary[0] < target_pos[1] < self.y_boundary[1]):
                target_pos[:2] = -target_pos[:2]
                back_pose = np.deg2rad([0.0, -180.0, 90.0, -90.0, 0.0, 0.0])
                starting_pose = np.deg2rad([-90.0, -80.0, -120.0, -70.0, 90.0, 0])
                self.robot_dual_control(rob1_pose=starting_pose, rob1_vel=0.75, rob1_acc=0.75,
                                        rob2_pose=back_pose, rob2_vel=0.75, rob2_acc=0.75)

                # : 로봇의 현재 위치 기록
                rob1_loc = self.rob1.getl()
                rob1_loc[0] = rob1_loc[0] - 0.0015
                rob1_preloc = copy.deepcopy(rob1_loc)
                # : 로봇의 x좌표먼저 이동
                rob1_preloc[0] = target_pos[0]
                self.rob1.movel(rob1_preloc, 1.0, 1.0)
                # : 타겟 좌표로 이동
                rob1_loc[:2] = target_pos[:2]
                rob1_loc[1] = rob1_loc[1] + 0.185
                self.rob1.movel(rob1_loc, 1.0, 1.0)

                robot_joint = self.rob1.getj()
                obj_angle, w, h = self.angle_detect(target_cls)
                if obj_angle < -120:
                    robot_joint[5] = np.deg2rad(obj_angle + np.rad2deg(robot_joint[0]) + 180.0)
                elif 120 < obj_angle:
                    robot_joint[5] = np.deg2rad(obj_angle + np.rad2deg(robot_joint[0]) - 180.0)
                else:
                    robot_joint[5] = np.deg2rad(obj_angle + np.rad2deg(robot_joint[0]))
                self.rob1.movej(robot_joint, acc=1.0, vel=1.0)

                env_show = self.env_show.copy()
                env_show = cv2.ellipse(env_show, (int(target_imgmean[1] / 2), int(target_imgmean[0] / 2)), (0, 10),
                                       obj_angle, 0, 360, (0, 128, 0))
                env_show = cv2.ellipse(env_show, (int(target_imgmean[1] / 2), int(target_imgmean[0] / 2)), (40, 0),
                                       obj_angle, 0, 360, (0, 255, 0))
                cv2.imshow("env_show", env_show)
                cv2.moveWindow("env_show", 0, 0)
                cv2.waitKey(2)

                robot_loc_after = self.rob1.getl()
                robot_loc_after[2] = copy.deepcopy(robot_loc_after[2]) - 0.248
                self.rob1.movel(robot_loc_after, 1.0, 1.0)
                self.gripper1.close_gripper()
                self.rob1.movel(rob1_loc, 1.0, 1.0)

                # 쓰레기통으로 이동
                target_pos[:2] = -bin_pos[:2]
                # rob1_loc[0] = target_pos[0] - 0.04 #좀더 왼쪽
                rob1_loc[0] = target_pos[0] - 0.07
                self.rob1.movel(rob1_loc, 1.0, 1.0)
                rob1_loc[1] = target_pos[1] + 0.23
                self.rob1.movel(rob1_loc, 0.5, 0.5)
                self.gripper1.open_gripper()
                self.rob1.movel(rob1_preloc, 1.0, 1.0)
                self.rob1.movej(starting_pose, 1.5, 1.5)

            else:
                logger.warning("%s is out of Safe Boundary" % RL_Obj_List[self.target_cls][0])
                self.obj_pos = None


    def open_drawer(self, drawer_pos=None):
        """
        Opens drawer with rob2.
        return
        :param drawer_cls: represents drawer class
        :param drawer_imgmean: drawer's ???
        :param drawer_pos: drawer's position
        ::
        """
        rob1_back_pose = np.deg2rad([0.0, 0.0, -90.0, -90.0, 0.0, 0.0])
        rob2_starting_pose = np.deg2rad([90.0, -100.0, 120.0, -110.0, -90.0, 0])
        rob2_move_path_j1 = np.deg2rad([31.60, -94.16, 114.86, -112.45, -90.63, -56.32])
        rob2_move_path_j2 = np.deg2rad([-65.53, -92.46, 115.08, -22.15, -245.37, 91.13])
        rob2_move_path_j3 = np.deg2rad([-65.56, -59.49, 130.47, -70.28, -246.74, 91.30])
        self.robot_dual_control(rob_total_pose=self.home, vel=1.0, acc=1.0)
        self.robot_dual_control(rob1_pose=rob1_back_pose, rob1_vel=0.75, rob1_acc=0.75,
                                rob2_pose=rob2_starting_pose, rob2_vel=0.75, rob2_acc=0.75)
        ##### Rob2: 서랍 열기 #####
        self.rob2.movej(rob2_move_path_j1, 1.0, 1.0)
        self.rob2.movej(rob2_move_path_j2, 1.0, 1.0)
        self.rob2.movej(rob2_move_path_j3, 1.0, 1.0)

        rob2_loc = self.rob2.getl()
        tmp_x = rob2_loc[0]
        rob2_loc[0:2] = drawer_pos[0:2]
        rob2_loc[0] = rob2_loc[0] + 0.03
        self.rob2.movel(rob2_loc, 1.0, 1.0)

        self.gripper2.close_gripper()
        rob2_loc[1] = rob2_loc[1] - 0.1    # 얼마나 drawer를 뺄지
        temp_loc = copy.deepcopy(rob2_loc)
        self.rob2.movel(rob2_loc, 1.0, 1.0)
        self.gripper2.open_gripper()

        rob2_loc[0] = tmp_x
        self.rob2.movel(rob2_loc, 1.0, 1.0)
        self.rob2.movej(rob2_move_path_j2, 1.0, 1.0)
        self.rob2.movej(rob2_move_path_j1, 1.0, 1.0)
        self.rob2.movej(rob2_starting_pose, 1.0, 1.0)

    def close_drawer(self, drawer_xyz):
        """
        Closes drawer.
        :param drawer_cls:
        :param drawer_imgmean:
        :param drawer_pos:
        :return:
        """
        rob1_back_pose = np.deg2rad([0.0, 0.0, -90.0, -90.0, 0.0, 0.0])
        rob2_starting_pose = np.deg2rad([90.0, -100.0, 120.0, -110.0, -90.0, 0])
        rob2_move_path_j1 = np.deg2rad([31.60, -94.16, 114.86, -112.45, -90.63, -56.32])
        rob2_move_path_j2 = np.deg2rad([-73.37, -91.49, 102.34, -100.34, -88.46, 15.72])
        rob2_move_path_j3 = np.deg2rad([-70.49, -70.50, 111.48, -135.33, -89.34, 22.18])
        self.robot_dual_control(rob_total_pose=self.home, vel=1.0, acc=1.0)
        self.robot_dual_control(rob1_pose=rob1_back_pose, rob1_vel=0.75, rob1_acc=0.75,
                                rob2_pose=rob2_starting_pose, rob2_vel=0.75, rob2_acc=0.75)

        # ##### Rob2: 서랍 닫기 #####
        self.rob2.movej(rob2_move_path_j1, 1.0, 1.0)
        self.rob2.movej(rob2_move_path_j2, 1.0, 1.0)
        self.rob2.movej(rob2_move_path_j3, 1.0, 1.0)
        rob2_loc = self.rob2.getl()
        rob2_loc[0] = drawer_xyz[0] + 0.03
        self.rob2.movel(rob2_loc, 1.0, 1.0)
        rob2_loc[1] = drawer_xyz[1] + 0.028
        self.rob2.movel(rob2_loc, 0.5, 0.5) # 서랍 넣기
        self.rob2.movej(rob2_move_path_j2, 1.0, 1.0)
        self.rob2.movej(rob2_move_path_j1, 1.0, 1.0)
        self.rob2.movej(rob2_starting_pose, 1.0, 1.0)

    def grasp_placing_drawer(self, target_cls, target_imgmean, obj_pos=None ):

        if obj_pos is None:
            logger.warning("target_pos is None")
            return
        else:
            target_pos = copy.deepcopy(obj_pos)  # --

            logger.info("target_pos : {}".format(target_pos))

            if (self.x_boundary[0] < target_pos[0] < self.x_boundary[1]) and (
                    self.y_boundary[0] < target_pos[1] < self.y_boundary[1]):

                rob1_starting_pose = np.deg2rad([-90.0, -80.0, -120.0, -70.0, 90.0, 0])
                rob1_back_pose = np.deg2rad([0.0, 0.0, -90.0, -90.0, 0.0, 0.0])
                rob2_starting_pose = np.deg2rad([90.0, -100.0, 120.0, -110.0, -90.0, 0])
                rob2_back_pose = np.deg2rad([0.0, -180.0, 90.0, -90.0, 0.0, 0.0])
                move_path_j1 = np.deg2rad([31.60, -94.16, 114.86, -112.45, -90.63, -56.32])
                move_path_j2 = np.deg2rad([-65.56, -59.49, 130.47, -70.28, -246.74, 91.30])
                self.robot_dual_control(rob_total_pose=self.home, vel=1.0, acc=1.0)
                self.robot_dual_control(rob1_pose=rob1_starting_pose, rob1_vel=0.75, rob1_acc=0.75,
                                        rob2_pose=rob2_back_pose, rob2_vel=0.75, rob2_acc=0.75)

                # ##### Rob1: 물체 잡고 서랍에 넣기 #####
                # : 로봇의 현재 위치 기록
                target_pos[:2] = -target_pos[:2]

                rob1_loc = self.rob1.getl()
                rob1_loc[0] = rob1_loc[0] - 0.0015
                rob1_preloc = copy.deepcopy(rob1_loc)
                # : 로봇의 x좌표먼저 이동
                rob1_preloc[0] = target_pos[0]
                self.rob1.movel(rob1_preloc, 1.0, 1.0)

                # : 타겟 좌표로 이동
                rob1_loc[:2] = target_pos[:2]
                rob1_loc[1] = rob1_loc[1] + 0.185
                self.rob1.movel(rob1_loc, 1.0, 1.0)

                robot_joint = self.rob1.getj()
                obj_angle, w, h = self.angle_detect(target_cls)
                if obj_angle < -120:
                    robot_joint[5] = np.deg2rad(obj_angle + np.rad2deg(robot_joint[0]) + 180.0)
                elif 120 < obj_angle:
                    robot_joint[5] = np.deg2rad(obj_angle + np.rad2deg(robot_joint[0]) - 180.0)
                else:
                    robot_joint[5] = np.deg2rad(obj_angle + np.rad2deg(robot_joint[0]))
                self.rob1.movej(robot_joint, acc=1.0, vel=1.0)

                env_show = self.env_show.copy()
                env_show = cv2.ellipse(env_show, (int(target_imgmean[1] / 2), int(target_imgmean[0] / 2)), (0, 10),
                                       obj_angle, 0, 360, (0, 128, 0))
                env_show = cv2.ellipse(env_show, (int(target_imgmean[1] / 2), int(target_imgmean[0] / 2)), (40, 0),
                                       obj_angle, 0, 360, (0, 255, 0))
                cv2.imshow("env_show", env_show)
                cv2.moveWindow("env_show", 0, 0)
                cv2.waitKey(2)

                robot_loc_after = self.rob1.getl()
                robot_loc_after[2] = copy.deepcopy(robot_loc_after[2]) - 0.248
                self.rob1.movel(robot_loc_after, 1.0, 1.0)

                self.gripper1.close_gripper()

                # rob2_loc[2] = rob2_loc[2] + 0.255
                self.rob1.movel(rob1_loc, 1.0, 1.0)

    def grasp_place_drawer_obj(self, drawer_pos=None):

        rob1_starting_pose = np.deg2rad([-90.0, -80.0, -120.0, -70.0, 90.0, 0])
        rob2_back_pose = np.deg2rad([0.0, -180.0, 90.0, -90.0, 0.0, 0.0])
        self.robot_dual_control(rob_total_pose=self.home, vel=1.0, acc=1.0)
        self.robot_dual_control(rob1_pose=rob1_starting_pose, rob1_vel=0.75, rob1_acc=0.75,
                                rob2_pose=rob2_back_pose, rob2_vel=0.75, rob2_acc=0.75)
        rob1_loc = self.rob1.getl()
        rob1_loc[0] = -drawer_pos[0]
        rob1_loc[1] = -drawer_pos[1] + 0.19
        rob1_loc[2] = -drawer_pos[2] + 0.06
        self.rob1.movel(rob1_loc, 1.0, 1.0)
        self.gripper1.open_gripper()

    # ---- ---- ---- ---- Pen ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
    def grasp_holder(self, target_cls, obj_pos=None):  # : Hand (왼쪽) 로봇으로 펜을 잡음
        if obj_pos is None:
            logger.warning("target_pos is None")
            return None
        else:
            target_pos = copy.deepcopy(obj_pos)
            logger.info("target_pos : {}".format(target_pos))

            if (self.x_boundary[0] < target_pos[0] < self.x_boundary[1]) and (
                    self.y_boundary[0] < target_pos[1] < self.y_boundary[1]):
                back_pose = np.deg2rad([0.0, 0.0, -90.0, -90.0, 0.0, 0.0])
                starting_pose = np.deg2rad([90.0, -90.0, 110.0, -110.0, -90.0, 0.0])
                self.robot_dual_control(rob_total_pose=self.home, vel=1.0, acc=1.0)
                self.robot_dual_control(rob1_pose=back_pose, rob1_vel=1.0, rob1_acc=1.0,
                                        rob2_pose=starting_pose, rob2_vel=0.75, rob2_acc=0.75)

                # : 로봇의 현재 위치 기록
                rob2_loc = self.rob2.getl()
                rob2_loc_org = copy.deepcopy(rob2_loc)
                rob2_loc[0] = rob2_loc[0] - 0.0015

                rob2_preloc = copy.deepcopy(rob2_loc)
                # : 로봇의 x좌표먼저 이동
                rob2_preloc[0] = target_pos[0]
                self.rob2.movel(rob2_preloc, 1.0, 1.0)

                # : 타겟 좌표로 이동
                rob2_loc[:2] = target_pos[:2]
                # rob2_loc[2] = rob2_loc[2]
                self.rob2.movel(rob2_loc, 1.0, 1.0)


                # 그립퍼 회전
                robot_joint = self.rob2.getj()
                obj_angle, w, h = self.angle_detect(target_cls)
                temp_angle = copy.deepcopy(robot_joint[5])
                gripper_angle = obj_angle + np.rad2deg(robot_joint[0])
                if gripper_angle < -120:
                    robot_joint[5] = np.deg2rad(obj_angle + np.rad2deg(robot_joint[0]) + 180.0)
                elif 120 < gripper_angle:
                    robot_joint[5] = np.deg2rad(obj_angle + np.rad2deg(robot_joint[0]) - 180.0)
                else:
                    robot_joint[5] = np.deg2rad(obj_angle + np.rad2deg(robot_joint[0]))
                self.rob2.movej(robot_joint, acc=1.0, vel=1.0)

                # obj_angle, w, h = self.angle_detect(target_cls)
                # temp_angle = copy.deepcopy(rob2_loc[4])
                # rob2_loc[4] = np.deg2rad(obj_angle)
                # # self.rob2.movej(robot_joint, acc=0.2, vel=0.2)

                robot_loc_after = self.rob2.getl()
                robot_loc_after[0] = copy.deepcopy(robot_loc_after[0]) + 0.03*cos(np.deg2rad(gripper_angle))
                robot_loc_after[1] = copy.deepcopy(robot_loc_after[1]) + 0.03*sin(np.deg2rad(gripper_angle))
                robot_loc_after[2] = copy.deepcopy(robot_loc_after[2]) - 0.205
                self.rob2.movel(robot_loc_after, 1.0, 1.0)

                self.gripper2.close_gripper()

                # rob2_loc[2] = rob2_loc[2] + 0.255
                self.rob2.movel(rob2_loc, 1.0, 1.0)
                holder_lpose = self.rob2.getl()

                self.rob2.movej(starting_pose, 1.0, 1.0)

                env_show = self.env_show.copy()  # x,            y
                start_point = [110, 600]
                placing_point = [160, 560]
                l = np.array(placing_point) - np.array(start_point)
                r = sqrt(math.pow(l[0], 2) + math.pow(l[1], 2))
                end_point = placing_point + 2 * l

                env_show = cv2.circle(env_show, (int(placing_point[0] / 2), int(placing_point[1] / 2)),
                                      2, (0, 255, 0), -1)
                env_show = cv2.circle(env_show, (int(start_point[0] / 2), int(start_point[1] / 2)),
                                      2, (255, 0, 0), -1)
                env_show = cv2.circle(env_show, (int(end_point[0] / 2), int(end_point[1] / 2)),
                                      2, (0, 0, 255), -1)

                env_show = cv2.ellipse(env_show, (int(160 / 2), int(560 / 2)), (0, int(r / 2)),
                                       90 - np.rad2deg(math.atan(-l[1] / l[0])), 0, 360, (0, 0, 255))

                
                cv2.imshow("env_show", env_show)
                cv2.moveWindow("env_show", 0, 0)
                cv2.waitKey(2)

                self.rob2.movel([-0.17825830789613528, -0.32658502248463084, 0.013097203443084932,
                                 0.01067246344967778, -3.0869902113673713, 0.031017305690963237], 0.2, 0.2)

                s_xyz, s_list, s_mean = self.get_obj_pos([start_point[1], start_point[0]], use_imgpoint=True)
                s_loc = self.rob2.getl()
                s_loc[0:2] = s_xyz[0:2]
                self.rob2.movel(s_loc + np.array([0, 0.05, 0, 0, 0, 0]), 0.2, 0.2)
                s_loc[2] = -0.0645
                self.rob2.movel(s_loc, 0.2, 0.2)
                e_xyz, e_list, e_mean = self.get_obj_pos([end_point[1], end_point[0]], use_imgpoint=True)
                e_loc = self.rob2.getl()
                e_loc[0:2] = e_xyz[0:2]
                s_loc[2] = -0.0645
                self.rob2.movel(e_loc, 0.2, 0.2)
                p_xyz, p_list, p_mean = self.get_obj_pos([placing_point[1], placing_point[0]], use_imgpoint=True)
                p_loc = self.rob2.getl()
                p_loc[0:2] = p_xyz[0:2]
                p_loc[2] = -0.0645
                self.rob2.movel(p_loc, 0.2, 0.2)

                self.gripper2.value_gripper(180)
                p_loc[2] = -0.0585 + 0.15
                self.rob2.movel(p_loc, 0.2, 0.2)
                self.gripper2.open_gripper()
                fixed_loc = self.rob2.getl()
                fixed_loc[0] = fixed_loc[0] - 0.038
                fixed_loc[2] = -0.0235
                self.rob2.movel(fixed_loc, 0.2, 0.2)

                fixed_loc = self.rob2.getl()
                fixed_loc[1] = fixed_loc[1] - 0.035
                fixed_loc[2] = -0.0585
                self.rob2.movel(fixed_loc, 0.2, 0.2)  

                vertical_loc = self.rob2.getl()
                vertical_loc[3:6] = [-2.6425233456565188e-05, -2.7206441140413893, -1.5707786178206657]
                self.rob2.movel(vertical_loc, 0.2, 0.2)
                vertical_loc[1] = vertical_loc[1] + 0.035
                self.rob2.movel(vertical_loc, 0.2, 0.2)

                self.gripper2.close_gripper()

                h_loc = self.rob2.getl()
                h_loc[2] = -0.0585 + 0.15
                self.rob2.movel(h_loc, 0.2, 0.2)

                self.rob2.movej(starting_pose, 1.0, 1.0)

                return copy.deepcopy(h_loc)

            else:
                print("%s is out of Safe Boundary" % RL_Obj_List[self.target_cls][0], file=sys.stderr)
                self.obj_pos = None
                return None

    def grasp_pen(self, target_cls, obj_pos=None):  # : Cam (오른쪽) 로봇으로 펜을 잡음                
        # 38mm
        if obj_pos is None:
            logger.warning("target_pos is None")
            return
        else:
            target_pos = copy.deepcopy(obj_pos)  # --
            logger.info("target_pos : {}".format(target_pos))

            if (self.x_boundary[0] < target_pos[0] < self.x_boundary[1]) and (
                    self.y_boundary[0] < target_pos[1] < self.y_boundary[1]):

                target_pos_rob2 = copy.deepcopy(target_pos)

                target_pos_rob1 = copy.deepcopy(target_pos)
                target_pos_rob1[0] = -target_pos_rob2[0]
                target_pos_rob1[1] = -target_pos_rob2[1] + 0.185

                back_pose = np.deg2rad([0.0, 0.0, -90.0, -90.0, 0.0, 0.0])
                starting_pose = np.deg2rad([90.0, -90.0, 110.0, -110.0, -90.0, 0.0])

                # : 로봇의 현재 위치 기록
                rob1_loc = self.rob1.getl()

                rob1_preloc = copy.deepcopy(rob1_loc)
                # : 로봇의 z좌표먼저 약간
                rob1_preloc[2] = target_pos_rob1[2] + 0.50
                self.rob1.movel(rob1_preloc, 0.5, 0.5)

                # : 타겟 좌표로 이동
                rob1_loc[:2] = target_pos_rob1[:2]
                rob1_loc[2] = target_pos_rob1[2] + 0.20
                # rob2_loc[2] = rob2_loc[2]
                self.rob1.movel(rob1_loc, 0.5, 0.5)

                robot_joint = self.rob1.getj()
                obj_angle, w, h = self.angle_detect(target_cls)
                temp_angle = copy.deepcopy(robot_joint[5])
                gripper_angle = obj_angle + np.rad2deg(robot_joint[0])
                if gripper_angle < -120:
                    robot_joint[5] = np.deg2rad(obj_angle + np.rad2deg(robot_joint[0]) + 180.0)
                elif 120 < gripper_angle:
                    robot_joint[5] = np.deg2rad(obj_angle + np.rad2deg(robot_joint[0]) - 180.0)
                else:
                    robot_joint[5] = np.deg2rad(obj_angle + np.rad2deg(robot_joint[0]))
                self.rob1.movej(robot_joint, acc=1.0, vel=1.0)

                robot1_loc_after = self.rob1.getl()
                robot1_loc_after[2] = copy.deepcopy(robot1_loc_after[2]) - 0.1975
                self.rob1.movel(robot1_loc_after, 0.5, 0.5)

                self.gripper1.close_gripper()

                self.rob1.movel(rob1_loc, 0.5, 0.5)

                self.rob1.movel(rob1_preloc, 0.5, 0.5)

                self.rob1.movej(back_pose, 1.0, 1.0)
            else:
                logger.warning("%s is out of Safe Boundary" % RL_Obj_List[self.target_cls][0])
                self.obj_pos = None

    def placing_toholder(self, h_loc):

        first_jpose = np.deg2rad([0.0, -90.0, -90.0, -90.0, 90.0, 90.0])
        self.robot_dual_control(rob1_pose=first_jpose, rob1_vel=0.75, rob1_acc=0.75,
                                rob2_pose=h_loc, rob2_vel=0.5, rob2_acc=0.5, is_r2_l=True)

        h2_loc = self.rob2.getl()
        h2_loc[1] = 0.039
        h2_loc[2] = 0.172
        second_jpose1 = [-0.019875828419820607, -1.3842194716082972, -1.9088614622699183,
                         -2.4182804266559046, 3.1394150257110596, 0.5713483095169067]

        self.robot_dual_control(rob1_pose=second_jpose1, rob1_vel=0.75, rob1_acc=0.75,
                                rob2_pose=h2_loc, rob2_vel=0.5, rob2_acc=0.5, is_r2_l=True)

        loc0 = self.rob1.getl()
        loc0[2] = loc0[2] - 0.15
        self.rob1.movel(loc0, 0.4, 0.4)
        self.gripper1.open_gripper()

        self.robot_dual_control(rob1_pose=first_jpose, rob1_vel=0.75, rob1_acc=0.75,
                                rob2_pose=h_loc, rob2_vel=0.5, rob2_acc=0.5, is_r2_l=True)

        starting_pose = np.deg2rad([90.0, -90.0, 110.0, -110.0, -90.0, 0.0])
        self.rob2.movej(starting_pose, 0.5, 0.5)

    def holder_toplace(self, h_loc):

        self.rob2.movel(h_loc, 0.5, 0.5)
        h_loc_ = copy.deepcopy(h_loc)
        h_loc_[2] = -0.058
        self.rob2.movel(h_loc_, 0.5, 0.5)
        self.gripper2.open_gripper()

        self.rob2.movel(h_loc, 0.5, 0.5)

        self.robot_dual_control(rob_total_pose=self.home, vel=1.0, acc=1.0)

    # ---- ---- ---- ---- Keyboard ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
    def grasp_placing_keyboard(self, target_cls, mean_xy):

        obj_angle, h, w = self.angle_detect(target_cls)
        obj_angle_rad = np.deg2rad(obj_angle)
        print("Target : {}, w : {}".format(RL_Obj_List[target_cls][0], w))
        print("w : {}".format(w))
        print("obj_angle+90 : {}".format(obj_angle + 90.0))

        # = img상의 pose 특정용 (일정 픽셀만큼 계산)
        pix_pos1 = mean_xy + [sin(obj_angle_rad + np.pi / 2.0) * (w / 2 + 30),
                              cos(obj_angle_rad + np.pi / 2.0) * (w / 2 + 30)]
        pix_pos2 = mean_xy + [sin(obj_angle_rad + np.pi / 2.0) * -(w / 2 + 30),
                              cos(obj_angle_rad + np.pi / 2.0) * -(w / 2 + 30)]

        pix_pos1_close = mean_xy + [sin(obj_angle_rad + np.pi / 2.0) * (w / 2 + 14),
                                    cos(obj_angle_rad + np.pi / 2.0) * (w / 2 + 14)]
        pix_pos2_close = mean_xy + [sin(obj_angle_rad + np.pi / 2.0) * -(w / 2 + 14),
                                    cos(obj_angle_rad + np.pi / 2.0) * -(w / 2 + 14)]

        # = 점 표시용
        color_seg = self.color_seg_img.copy()
        color_seg = cv2.circle(color_seg, (int(pix_pos1[1]), int(pix_pos1[0])), 10, (255, 0, 0), -1)
        color_seg = cv2.circle(color_seg, (int(pix_pos2[1]), int(pix_pos2[0])), 10, (255, 0, 0), -1)

        color_seg = cv2.circle(color_seg, (int(pix_pos1_close[1]), int(pix_pos1_close[0])), 10, (0, 255, 0), -1)
        color_seg = cv2.circle(color_seg, (int(pix_pos2_close[1]), int(pix_pos2_close[0])), 10, (0, 255, 0), -1)

        colorseg_show = cv2.resize(color_seg, (int(color_seg.shape[1] / 2), int(color_seg.shape[0] / 2)))
        cv2.imshow("colorseg_show", colorseg_show)
        cv2.moveWindow("colorseg_show", 0, 390)
        cv2.waitKey(2)

        env_show = self.env_show.copy()
        env_show = cv2.circle(env_show, (int(pix_pos1[1] / 2), int(pix_pos1[0] / 2)), 3, (255, 0, 0), -1)
        env_show = cv2.circle(env_show, (int(pix_pos2[1] / 2), int(pix_pos2[0] / 2)), 3, (255, 0, 0), -1)

        env_show = cv2.circle(env_show, (int(pix_pos1_close[1] / 2), int(pix_pos1_close[0] / 2)), 3, (0, 255, 0),
                              -1)
        env_show = cv2.circle(env_show, (int(pix_pos2_close[1] / 2), int(pix_pos2_close[0] / 2)), 3, (0, 255, 0),
                              -1)

        env_show = cv2.ellipse(env_show, (int(mean_xy[1] / 2), int(mean_xy[0] / 2)), (0, 121), obj_angle, 0, 360,
                               (0, 0, 255))
        env_show = cv2.ellipse(env_show, (int(mean_xy[1] / 2), int(mean_xy[0] / 2)), (20, 0), obj_angle, 0, 360,
                               (0, 0, 128))

        cv2.imshow("env_show", env_show)
        cv2.moveWindow("env_show", 0, 0)
        cv2.waitKey(2)
        # : 이후 xy, 로봇 미지정 상태의 이미지상 좌표

        # : [y, x]
        if pix_pos1[1] < pix_pos2[1]:
            obj_pos1 = self.global_cam.pxl2xyz([1], pix_pos2)  # cam
            obj_pos1_close = self.global_cam.pxl2xyz([1], pix_pos2_close)

            obj_pos2 = self.global_cam.pxl2xyz([1], pix_pos1)  # hand
            obj_pos2_close = self.global_cam.pxl2xyz([1], pix_pos1_close)

            if pix_pos1[0] > pix_pos2[0]:
                prior = 2
            else:
                prior = 1
        else:
            obj_pos1 = self.global_cam.pxl2xyz([1], pix_pos1)
            obj_pos1_close = self.global_cam.pxl2xyz([1], pix_pos2_close)

            obj_pos2 = self.global_cam.pxl2xyz([1], pix_pos2)
            obj_pos2_close = self.global_cam.pxl2xyz([1], pix_pos1_close)

            if pix_pos1[0] > pix_pos2[0]:
                prior = 1
            else:
                prior = 2

        # : 이후 xyz, obj_pos1 = 오른쪽(cam), obj_pos2 = 왼쪽(hand)
        if (self.x_boundary[0] < obj_pos1[0] < self.x_boundary[1]) and \
                (self.y_boundary[0] < obj_pos1[1] < self.y_boundary[1]) and \
                (self.x_boundary[0] < obj_pos2[0] < self.x_boundary[1]) and \
                (self.y_boundary[0] < obj_pos2[1] < self.y_boundary[1]):
            goal1 = np.array(obj_pos1)
            goal1 = self.robot_action_mod(goal1)
            close1 = np.array(obj_pos1_close)
            close1 = self.robot_action_mod(close1)

            goal2 = np.array(obj_pos2)
            close2 = np.array(obj_pos2_close)
        else:
            logger.warning("%s is out of Safe Boundary" % RL_Obj_List[target_cls][0])
            self.obj_pos = None
            return
        # : 이후 xyz, goal1 = 오른쪽(cam), goal2 = 왼쪽(hand)
        self.robot_dual_control(rob1_pose=self.initial_pose1, rob1_vel=0.75, rob1_acc=0.75,
                                rob2_pose=self.initial_pose2, rob2_vel=0.75, rob2_acc=0.75)

        rob_pos1 = self.rob1.getl()
        rob_pos1[:2] = goal1[:2]
        rob_pos1[2] = self.z_lift

        rob_pos2 = self.rob2.getl()
        rob_pos2[:2] = goal2[:2]
        rob_pos2[2] = self.z_lift

        # 아아.... 귀차나
        if prior == 1:
            self.rob1.movel(rob_pos1, 0.5, 0.5, False)
            self.rob1._wait_for_move(rob_pos1)
            self.rob2.movel(rob_pos2, 0.5, 0.5, False)
            self.rob2._wait_for_move(rob_pos2)
        else:
            self.rob2.movel(rob_pos2, 0.5, 0.5, False)
            self.rob2._wait_for_move(rob_pos2)
            self.rob1.movel(rob_pos1, 0.5, 0.5, False)
            self.rob1._wait_for_move(rob_pos1)

        rotated_joint_position1 = np.append(self.rob1.getj()[:-1], self.rob1.getj()[-1] + obj_angle_rad)
        rotated_pose1 = self.solve_FK(rotated_joint_position1)
        rotated_joint_position2 = np.append(self.rob2.getj()[:-1], self.rob2.getj()[-1] + obj_angle_rad)
        rotated_pose2 = self.solve_FK(rotated_joint_position2)

        rob_pos1[3:] = rotated_pose1[3:]
        rob_pos2[3:] = rotated_pose2[3:]

        self.rob1.movel(rob_pos1, 0.5, 0.5, False)
        self.rob2.movel(rob_pos2, 0.5, 0.5, False)
        self.rob1._wait_for_move(rob_pos1)
        self.rob2._wait_for_move(rob_pos2)

        self.action_gripper_both(80)

        rob_pos1[2] = - 0.11635
        rob_pos2[2] = - 0.11635

        self.rob1.movel(rob_pos1, 0.1, 0.1, False)
        self.rob2.movel(rob_pos2, 0.1, 0.1, False)
        self.rob1._wait_for_move(rob_pos1)
        self.rob2._wait_for_move(rob_pos2)

        rob_pos1[:2] = close1[:2]
        rob_pos2[:2] = close2[:2]

        # = may the Force be with you
        self.rob1.movel(rob_pos1, 0.005, 0.01, False)
        self.rob2.movel(rob_pos2, 0.005, 0.01, False)
        self.rob1._wait_for_move(rob_pos1, force_thr=True)
        self.rob2._wait_for_move(rob_pos2, force_thr=True)

        rob_pos1 = self.rob1.getl()
        rob_pos2 = self.rob2.getl()
        xy_pos1 = np.array([rob_pos1[0], rob_pos1[1]])
        xy_pos2 = np.array([rob_pos2[0], rob_pos2[1]])
        xy_pos1 = xy_pos1 + [sin(obj_angle_rad + np.pi / 2.0) * - 0.0014, cos(obj_angle_rad + np.pi / 2.0) * - 0.0014]
        xy_pos2 = xy_pos2 + [sin(obj_angle_rad + np.pi / 2.0) * - 0.0014, cos(obj_angle_rad + np.pi / 2.0) * - 0.0014]
        rob_pos1[:2] = xy_pos1
        rob_pos2[:2] = xy_pos2
        self.rob1.movel(rob_pos1, 0.05, 0.05, False)
        self.rob2.movel(rob_pos2, 0.05, 0.05, False)
        self.rob1._wait_for_move(rob_pos1)
        self.rob2._wait_for_move(rob_pos2)

        rob_pos1 = self.rob1.getl()
        rob_pos2 = self.rob2.getl()
        rob_pos1[2] = -0.096
        rob_pos2[2] = -0.096

        self.rob1.movel(rob_pos1, 0.05, 0.05, False)
        self.rob2.movel(rob_pos2, 0.05, 0.05, False)
        self.rob1._wait_for_move(rob_pos1)
        self.rob2._wait_for_move(rob_pos2)

        # 회전시키기
        rob_pos1 = self.rob1.getl()
        rob_pos2 = self.rob2.getl()
        rob_pos1_j = self.rob1.getj()
        rob_pos2_j = self.rob2.getj()

        rob_xy1 = [-rob_pos1[0], -rob_pos1[1] + 0.185]
        rob_xy2 = [rob_pos2[0], rob_pos2[1]]
        center = np.array([(rob_xy1[0] + rob_xy2[0]) / 2, (rob_xy1[1] + rob_xy2[1]) / 2])
        width = sqrt((rob_xy1[0] - rob_xy2[0]) ** 2 + (rob_xy1[1] - rob_xy2[1]) ** 2)

        #         위 0
        #  -90 왼 ㅇ 오 90
        #         아
        if -90 <= obj_angle:
            angle_list = [angle for angle in range(int(obj_angle), - 90, -1)]
        else:
            angle_list = [angle for angle in range(int(obj_angle), - 90, 1)]
        angle_list.append(-90)

        move_list1 = []
        move_list2 = []
        for angle in angle_list:
            rob_pos1_mod = rob_pos1.copy()
            rob_pos2_mod = rob_pos2.copy()
            angle_rad = np.deg2rad(angle)
            rot_xy1 = center + [sin(angle_rad + np.pi / 2.0) * (width / 2),
                                cos(angle_rad + np.pi / 2.0) * (width / 2)]
            rot_xy1 = self.robot_action_mod(rot_xy1)
            rot_xy2 = center + [sin(angle_rad + np.pi / 2.0) * -(width / 2),
                                cos(angle_rad + np.pi / 2.0) * -(width / 2)]
            rob_pos1_mod[:2] = rot_xy1
            rob_pos2_mod[:2] = rot_xy2

            rot = np.deg2rad(angle - obj_angle)
            rot1 = np.append(rob_pos1_j[:-1], rob_pos1_j[-1] + rot)
            rot2 = np.append(rob_pos2_j[:-1], rob_pos2_j[-1] + rot)
            rot_pose1 = self.solve_FK(rot1)
            rot_pose2 = self.solve_FK(rot2)
            rob_pos1_mod[3:] = rot_pose1[3:]
            rob_pos2_mod[3:] = rot_pose2[3:]

            move_list1.append(rob_pos1_mod)
            move_list2.append(rob_pos2_mod)

        self.rob1.movels(move_list1, 0.5, 0.8, radius=0.002, wait=False)
        self.rob2.movels(move_list2, 0.5, 0.8, radius=0.002, wait=False)
        self.rob1._wait_for_move(move_list1[-1], timeout=10.0)
        self.rob2._wait_for_move(move_list2[-1], timeout=10.0)

        rob_pos1 = self.rob1.getl()
        rob_pos2 = self.rob2.getl()
        goal_position = np.array([-0.683, 0.282])
        goal_angle = np.deg2rad(-90.0)
        rot_xy1 = goal_position + [sin(goal_angle + np.pi / 2.0) * (width / 2),
                                   cos(goal_angle + np.pi / 2.0) * (width / 2)]
        rot_xy1 = self.robot_action_mod(rot_xy1)
        rot_xy2 = goal_position + [sin(goal_angle + np.pi / 2.0) * -(width / 2),
                                   cos(goal_angle + np.pi / 2.0) * -(width / 2)]
        rob_pos1[:2] = rot_xy1
        rob_pos2[:2] = rot_xy2

        self.rob2.movel(rob_pos2, 0.05, 0.05, False)
        self.rob1.movel(rob_pos1, 0.05, 0.05, False)
        self.rob1._wait_for_move(rob_pos1, threshold=0.03, timeout=10.0)
        self.rob2._wait_for_move(rob_pos2, threshold=0.03, timeout=10.0)

        rob_pos1[2] = - 0.108
        rob_pos2[2] = - 0.108

        self.rob1.movel(rob_pos1, 0.05, 0.05, False)
        self.rob2.movel(rob_pos2, 0.05, 0.05, False)
        self.rob1._wait_for_move(rob_pos1, timeout=10.0)
        self.rob2._wait_for_move(rob_pos2, timeout=10.0)

        rot_xy1_open = goal_position + [sin(goal_angle + np.pi / 2.0) * (width / 2 + 0.05),
                                        cos(goal_angle + np.pi / 2.0) * (width / 2 + 0.05)]
        rot_xy1_open = self.robot_action_mod(rot_xy1_open)
        rot_xy2_open = goal_position + [sin(goal_angle + np.pi / 2.0) * -(width / 2 + 0.05),
                                        cos(goal_angle + np.pi / 2.0) * -(width / 2 + 0.05)]
        rob_pos1[:2] = rot_xy1_open
        rob_pos2[:2] = rot_xy2_open

        self.rob1.movel(rob_pos1, 0.1, 0.1, False)
        self.rob2.movel(rob_pos2, 0.1, 0.1, False)
        self.rob1._wait_for_move(rob_pos1, timeout=10.0)
        self.rob2._wait_for_move(rob_pos2, timeout=10.0)

        rob_pos1[2] = self.z_lift
        rob_pos2[2] = self.z_lift

        self.rob1.movel(rob_pos1, 0.1, 0.1, False)
        self.rob2.movel(rob_pos2, 0.1, 0.1, False)
        self.rob1._wait_for_move(rob_pos1, timeout=10.0)
        self.rob2._wait_for_move(rob_pos2, timeout=10.0)

        self.reset()

    def grasp_open_pen_lid(self, target_cls, mean_xy, obj_pos=None):
        """
        Grab pen by rob2, and take the lid by rob1.
        :param mean_xy:
        :return:
        """
        if obj_pos is None:
            logging.warning("sys : target_pos is None")
            return
        else:
            target_pos = copy.deepcopy(obj_pos)
            logging.info("target_pos : {}".format(target_pos))

            if (self.x_boundary[0] < target_pos[0] < self.x_boundary[1]) and (
                    self.y_boundary[0] < target_pos[1] < self.y_boundary[1]):
                rob1_starting_pose = np.deg2rad([-90.0, -80.0, -120.0, -70.0, 90.0, 0])
                rob1_back_pose = np.deg2rad([0.0, 0.0, -90.0, -90.0, 0.0, 0.0])
                rob2_starting_pose = np.deg2rad([90.0, -100.0, 120.0, -110.0, -90.0, 0])
                rob2_back_pose = np.deg2rad([0.0, -180.0, 90.0, -90.0, 0.0, 0.0])
                placing_pose = np.deg2rad([90.0, -120.0, 140.0, -110.0, -90.0, 0])
                self.robot_dual_control(rob1_pose=rob1_back_pose, rob1_vel=1.0, rob1_acc=1.0,
                                        rob2_pose=rob2_starting_pose, rob2_vel=0.75, rob2_acc=0.75)
                # STEP 1: Grap pen by rob2
                # : 로봇의 현재 위치 기록
                rob2_loc = self.rob2.getl()
                rob2_loc[0] = rob2_loc[0] - 0.0015

                rob2_preloc = copy.deepcopy(rob2_loc)
                # : 로봇의 x좌표먼저 이동
                rob2_preloc[0] = target_pos[0]
                self.rob2.movel(rob2_preloc, 0.5, 0.5)

                # : 타겟 좌표로 이동
                rob2_loc[:2] = target_pos[:2]
                self.rob2.movel(rob2_loc, 0.5, 0.5)

                robot_joint = self.rob2.getj()
                obj_angle, w, h = self.angle_detect(target_cls)
                temp_angle = copy.deepcopy(robot_joint[5])
                gripper_angle = obj_angle + np.rad2deg(robot_joint[0])
                if obj_angle < -120:
                    robot_joint[5] = np.deg2rad(obj_angle + np.rad2deg(robot_joint[0]) + 180.0)
                elif 120 < obj_angle:
                    robot_joint[5] = np.deg2rad(obj_angle + np.rad2deg(robot_joint[0]) - 180.0)
                else:
                    robot_joint[5] = np.deg2rad(obj_angle + np.rad2deg(robot_joint[0]))
                self.rob2.movej(robot_joint, acc=1.0, vel=1.0)
                robot_loc_after = self.rob2.getl()
                #robot_loc_after[2] = copy.deepcopy(robot_loc_after[2]) - 0.246
                robot_loc_after[2] = copy.deepcopy(robot_loc_after[2]) - 0.243
                self.rob2.movel(robot_loc_after, 0.5, 0.5)
                self.gripper2.close_gripper()

                # rob2_loc[2] = rob2_loc[2] + 0.255
                self.rob2.movel(rob2_loc, 0.5, 0.5)

                #rob2 조인트잡기
                rob2_grapping_joint1 = np.deg2rad([-5.34, -82.00, 71.91, 7.45, 172.72, -2.43])
                # TODO: 좀 더 효율적으로 움직이게 joint 설정
                self.rob2.movej(rob2_grapping_joint1, 0.5, 0.5)
                #rob1 조인트잡기
                self.rob1.movej(self.home, 0.5, 0.5)
                self.rob1.movej(rob1_starting_pose, 0.5, 0.5)
                rob1_grapping_joint = np.deg2rad([-15.20, -104.14, -69.34, -186.35, 170.98, 0])
                self.rob1.movej(rob1_grapping_joint, 0.5, 0.5)
                self.gripper1.close_gripper()
                #rob2 펜 끝 잡기
                self.gripper2.open_gripper()
                rob2_grapping_joint2 = np.deg2rad([-5.51, -78.93, 61.44, 17.16, 172.55, 0])
                self.rob2.movej(rob2_grapping_joint2, 0.5, 0.5)
                self.gripper2.close_gripper()
                #rob2 위로 - 뚜껑 열기
                rob2_loc = self.rob2.getl()
                rob2_loc[2] = rob2_loc[2] + 0.05
                self.rob2.movel(rob2_loc, 0.5, 0.5)
                #rob2 아래로 - 뚜껑 닫기
                rob2_loc[2] = rob2_loc[2] - 0.052
                self.rob2.movel(rob2_loc, 0.5, 0.5)
                #둘다 그리퍼 열어서 떨구기
                self.gripper1.open_gripper()
                self.gripper2.open_gripper()
                #배콤
                self.robot_dual_control(rob_total_pose=self.home, vel=1.0, acc=1.0)
                self.robot_dual_control(rob1_pose=rob1_back_pose, rob1_vel=1.0, rob1_acc=1.0,
                                        rob2_pose=rob2_starting_pose, rob2_vel=0.75, rob2_acc=0.75)
                

    def grasp_open_bottle_lid(self, target_cls, mean_xy, obj_pos=None):
        """
        Grab pen by rob2, and take the lid by rob1.
        :param mean_xy:
        :return:
        """
        if obj_pos is None:
            logging.warning("sys : target_pos is None")
            return
        else:
            target_pos = copy.deepcopy(obj_pos)
            logging.info("target_pos : {}".format(target_pos))

            if (self.x_boundary[0] < target_pos[0] < self.x_boundary[1]) and (
                    self.y_boundary[0] < target_pos[1] < self.y_boundary[1]):
                rob1_starting_pose = np.deg2rad([-90.0, -80.0, -120.0, -70.0, 90.0, 0])
                rob1_back_pose = np.deg2rad([0.0, 0.0, -90.0, -90.0, 0.0, 0.0])
                rob2_starting_pose = np.deg2rad([90.0, -100.0, 120.0, -110.0, -90.0, 0])
                rob2_back_pose = np.deg2rad([0.0, -180.0, 90.0, -90.0, 0.0, 0.0])
                placing_pose = np.deg2rad([90.0, -120.0, 140.0, -110.0, -90.0, 0])
                self.robot_dual_control(rob1_pose=rob1_back_pose, rob1_vel=1.0, rob1_acc=1.0,
                                        rob2_pose=rob2_starting_pose, rob2_vel=0.75, rob2_acc=0.75)
                # STEP 1: Grap bottle lid by rob2
                # 로봇의 현재 위치 기록
                rob2_loc = self.rob2.getl()
                rob2_loc[0] = rob2_loc[0] - 0.0015
                rob2_preloc = copy.deepcopy(rob2_loc)
                # 로봇의 x좌표먼저 이동
                rob2_preloc[0] = target_pos[0] + 0.05
                self.rob2.movel(rob2_preloc, 0.5, 0.5)

                # 타겟 좌표로 이동
                rob2_loc[0] = rob2_preloc[0]
                rob2_loc[1] = target_pos[1] + 0.01
                self.rob2.movel(rob2_loc, 0.5, 0.5)
                robot_loc_after = self.rob2.getl()
                robot_loc_after[2] -= 0.05
                self.rob2.movel(robot_loc_after, 0.5, 0.5)
                self.gripper2.close_gripper()
                self.rob2.movel(rob2_loc, 0.5, 0.5)

                # movel 함수로 중앙으로 이동
                rob2_center_joint = np.deg2rad([-27.08, -76.78, 103.95, -117.91, -89.17, 0.00])
                self.rob2.movej(rob2_center_joint, 0.3, 0.3)
                temp_loc = self.rob2.getl()

                # STEP 2: Grap center point of bottle by rob1
                rob1_mid_pos2 = np.deg2rad([-39.00, -113.35, -122.70, -114.19, -177.21, 9.52])
                rob1_holding_pos = np.deg2rad([-3.01, -121.97, -123.85, -112.72, -140.44, 0.34])
                self.rob1.movej(self.home, 0.6, 0.6)
                self.rob1.movej(rob1_mid_pos2, 0.4, 0.4)
                self.rob1.movej(rob1_holding_pos, 0.5, 0.5)
                self.gripper1.close_gripper()

                # STEP 3: Open the lid of bottle by rob1
                rob2_open_joint = np.deg2rad([-27.08, -76.98, 103.70, -117.45, -89.16, -240.00])
                self.rob2.movej(rob2_open_joint, 0.5, 0.5)  # 뚜껑 열기

                rob2_loc = self.rob2.getl()
                rob2_loc[2] += 0.05
                self.rob2.movel(rob2_loc, 0.4, 0.4)
                rob2_loc[2] -= 0.051
                self.rob2.movel(rob2_loc, 0.4, 0.4)
                self.rob2.movej(rob2_center_joint, 0.5, 0.5)  # 뚜껑 닫기

                self.gripper1.open_gripper()
                self.rob1.movej(rob1_mid_pos2, 0.4, 0.4)
                self.rob1.movej(self.home, 0.5, 0.5)
                self.rob1.movej(rob1_back_pose, 0.5, 0.5)

                rob2_path1 = np.deg2rad([31.63, -104.47, 92.88, -80.02, -88.11, 8.55])
                rob2_path2 = np.deg2rad([69.86, -124.32, 121.09, -88.53, -88.23, 8.60])
                self.rob2.movej(rob2_path1, 0.7, 0.7)
                self.rob2.movej(rob2_path2, 0.5, 0.5)
                self.gripper2.open_gripper()
                self.rob2.movej(rob2_starting_pose, 0.5, 0.5)


    def grasp_moving_cleaner(self, target_cls, target_imgmean, obj_pos=None):

        if obj_pos is None:
            logger.warning("target_pos is None")
            return
        else:
            target_pos = copy.deepcopy(obj_pos)  # --
            grasp_open_pen_lid("target_pos : {}".format(target_pos))

            if (self.x_boundary[0] < target_pos[0] < self.x_boundary[1]) and (
                    self.y_boundary[0] < target_pos[1] < self.y_boundary[1]):

                back_pose = np.deg2rad([0.0, 0.0, -90.0, -90.0, 0.0, 0.0])
                starting_pose = np.deg2rad([90.0, -100.0, 120.0, -110.0, -90.0, 0])
                placing_pose = np.deg2rad([90.0, -120.0, 140.0, -110.0, -90.0, 0])
                self.robot_dual_control(rob1_pose=back_pose, rob1_vel=1.0, rob1_acc=1.0,
                                        rob2_pose=starting_pose, rob2_vel=0.75, rob2_acc=0.75)

                # : 로봇의 현재 위치 기록
                rob2_loc = self.rob2.getl()
                rob2_loc[0] = rob2_loc[0] - 0.0015

                rob2_preloc = copy.deepcopy(rob2_loc)
                # : 로봇의 x좌표먼저 이동
                rob2_preloc[0] = target_pos[0]
                self.rob2.movel(rob2_preloc, 0.5, 0.5)

                # : 타겟 좌표로 이동
                rob2_loc[:2] = target_pos[:2]
                self.rob2.movel(rob2_loc, 0.5, 0.5)

                robot_joint = self.rob2.getj()
                obj_angle, w, h = self.angle_detect(target_cls)
                temp_angle = copy.deepcopy(robot_joint[5])
                gripper_angle = obj_angle + np.rad2deg(robot_joint[0])
                if obj_angle < -120:
                    robot_joint[5] = np.deg2rad(obj_angle + np.rad2deg(robot_joint[0]) + 180.0)
                elif 120 < obj_angle:
                    robot_joint[5] = np.deg2rad(obj_angle + np.rad2deg(robot_joint[0]) - 180.0)
                else:
                    robot_joint[5] = np.deg2rad(obj_angle + np.rad2deg(robot_joint[0]))
                self.rob2.movej(robot_joint, acc=1.0, vel=1.0)

                env_show = self.env_show.copy()
                env_show = cv2.ellipse(env_show, (int(target_imgmean[1] / 2), int(target_imgmean[0] / 2)), (0, 10),
                                       obj_angle, 0, 360, (0, 128, 0))
                env_show = cv2.ellipse(env_show, (int(target_imgmean[1] / 2), int(target_imgmean[0] / 2)), (40, 0),
                                       obj_angle, 0, 360, (0, 255, 0))
                cv2.imshow("env_show", env_show)
                cv2.moveWindow("env_show", 0, 0)
                cv2.waitKey(2)

                robot_loc_after = self.rob2.getl()
                robot_loc_after[2] = copy.deepcopy(robot_loc_after[2]) - 0.246
                self.rob2.movel(robot_loc_after, 0.5, 0.5)

                self.gripper2.close_gripper()

                # rob2_loc[2] = rob2_loc[2] + 0.255
                self.rob2.movel(rob2_loc, 0.5, 0.5)
                moving_path_start = np.deg2rad([-64.75, -61.01, 118.68, -147.91, -88.81, -154.27])
                self.rob2.movej(moving_path_start, 0.5, 0.5)
                rob2_loc = self.rob2.getl()
                rob2_loc[0] -= 0.18
                self.rob2.movel(rob2_loc, 0.5, 0.5)
                rob2_loc[1] -= 0.19
                self.rob2.movel(rob2_loc, 0.5, 0.5)
                rob2_loc[0] += 0.18
                self.rob2.movel(rob2_loc, 0.5, 0.5)
                rob2_loc[1] -= 0.19
                self.rob2.movel(rob2_loc, 0.5, 0.5)
                rob2_loc[0] -= 0.4
                self.rob2.movel(rob2_loc, 0.5, 0.5)
                rob2_loc[1] -= 0.11
                self.rob2.movel(rob2_loc, 0.5, 0.5)
                rob2_loc[0] += 0.2
                self.rob2.movel(rob2_loc, 0.5, 0.5)
                rob2_loc[1] -= 0.11
                self.rob2.movel(rob2_loc, 0.5, 0.5)
                rob2_loc[0] -= 0.2
                self.rob2.movel(rob2_loc, 0.5, 0.5)

                self.gripper2.open_gripper()
                rob2_loc[2] += 0.1
                self.rob2.movel(rob2_loc, 0.5, 0.5)
                self.reset()

            else:
                logger.warning("%s is out of Safe Boundary" % RL_Obj_List[self.target_cls][0])
                self.obj_pos = None


if __name__ == '__main__':
    socket_ip1 = "192.168.10.29"  # 오른쪽
    socket_ip2 = "192.168.10.52"  # 왼쪽
    rob = Robot(socket_ip1, socket_ip2)

    rob.reset()
