"""
RL Data Collector
latest Ver.180405
"""

# Robot
from urx import robotiq_two_finger_gripper
import socket
import urx
import time
import numpy as np
import sys
import copy

import serial

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


class Robot_util:
    def __init__(self, socket_ip):
        print("-->> trying to connect robot")
        while True:
            try:
                # Robot & Gripper
                self.rob = urx.Robot(socket_ip, use_rt=True)
                self.gripper = robotiq_two_finger_gripper.Robotiq_Two_Finger_Gripper(self.rob)
                self.gripper.open_gripper()
                # self.urscript_mod = self.gripper._get_new_urscript()
                # a = self.urscript_mod._get_gripper_status_mod()

                # Dashboard Control
                self.Dashboard_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.Dashboard_socket.connect((socket_ip, 29999))
                self._program_send("")

                # # Tray, Bluetooth
                # self.bluetooth = serial.Serial(serial_num, 9600, timeout=1)  # Tray # pre-move

                # Robot Dynamics & Kinematics Parameters
                self.ur5_a = [0, -0.425, -0.39225, 0, 0, 0]
                self.ur5_d = [0.089159, 0, 0, 0.10915, 0.09465, 0.0823]
                self.alp = [np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0]
                break
            except:
                print("..>> connecting failed. trying to reconnect robot {}...".format(socket_ip))

        print('\x1b[1;31;0m' + "-->>Robot util _ {} Ready.".format(socket_ip) + '\x1b[0m', file=sys.stderr)

    def chk_STA(self):  # : 20191118
        self.gripper.chk_sta_gripper()

    def _program_send(self, cmd):
        self.Dashboard_socket.send(cmd.encode())
        return self.Dashboard_socket.recv(1024).decode("utf-8")  # received byte data to string

    # def shuffle(self):
    #     self.bluetooth.write("2".encode())

    # def _AH(self, n, th, c):
    #     T_a = np.matrix(np.identity(4), copy=False)
    #     T_a[0, 3] = self.ur5_a[n - 1]
    #     T_d = np.matrix(np.identity(4), copy=False)
    #     T_d[2, 3] = self.ur5_d[n - 1]
    #
    #     Rzt = np.matrix([[cos(th[n - 1, c]), -sin(th[n - 1, c]), 0, 0],
    #                      [sin(th[n - 1, c]), cos(th[n - 1, c]), 0, 0],
    #                      [0, 0, 1, 0],
    #                      [0, 0, 0, 1]], copy=False)
    #
    #     Rxa = np.matrix([[1, 0, 0, 0],
    #                      [0, cos(self.alp[n - 1]), -sin(self.alp[n - 1]), 0],
    #                      [0, sin(self.alp[n - 1]), cos(self.alp[n - 1]), 0],
    #                      [0, 0, 0, 1]], copy=False)
    #
    #     A_i = T_d * Rzt * T_a * Rxa
    #
    #     return A_i

    # def HTrans(self, th, c):
    #     A_1 = self._AH(1, th, c)
    #     A_2 = self._AH(2, th, c)
    #     A_3 = self._AH(3, th, c)
    #     A_4 = self._AH(4, th, c)
    #     A_5 = self._AH(5, th, c)
    #     A_6 = self._AH(6, th, c)
    #
    #     T_06 = A_1 * A_2 * A_3 * A_4 * A_5 * A_6
    #
    #     return T_06

    # def solve_FK(self, th):
    #     T = np.eye(4)
    #     return_T = np.eye(4)
    #
    #     for i in range(6):
    #         T[0, 0] = cos(th[i])
    #         T[0, 1] = -sin(th[i]) * cos(self.alp[i])
    #         T[0, 2] = sin(th[i]) * sin(self.alp[i])
    #         T[0, 3] = self.ur5_a[i] * cos(th[i])
    #
    #         T[1, 0] = sin(th[i])
    #         T[1, 1] = cos(th[i]) * cos(self.alp[i])
    #         T[1, 2] = -cos(th[i]) * sin(self.alp[i])
    #         T[1, 3] = self.ur5_a[i] * sin(th[i])
    #
    #         T[2, 0] = 0
    #         T[2, 1] = sin(self.alp[i])
    #         T[2, 2] = cos(self.alp[i])
    #         T[2, 3] = self.ur5_d[i]
    #
    #         T[3, 0] = 0
    #         T[3, 1] = 0
    #         T[3, 2] = 0
    #         T[3, 3] = 1
    #
    #         return_T = (return_T @ T)
    #
    #     pose_vector = m3d.Transform(return_T).pose_vector
    #     return pose_vector

    # def solve_IK(self, desired_pos, q):  # T60
    #     desired_pos = m3d.Transform(desired_pos).matrix
    #
    #     th = np.matrix(np.zeros((6, 8)))
    #     P_05 = desired_pos * \
    #            np.matrix([0, 0, -self.ur5_d[5], 1]).T - np.matrix([0, 0, 0, 1]).T
    #
    #     # **** theta1 ****
    #     psi = atan2(P_05[2 - 1, 0], P_05[1 - 1, 0])
    #     phi = acos(self.ur5_d[3] / sqrt(P_05[2 - 1, 0] * \
    #                                     P_05[2 - 1, 0] + P_05[1 - 1, 0] * P_05[1 - 1, 0]))
    #     # The two solutions for theta1 correspond to the shoulder
    #     # being either left or right
    #     th[0, 0:4] = pi / 2 + psi + phi
    #     th[0, 4:8] = pi / 2 + psi - phi
    #     th = th.real
    #
    #     # **** theta5 ****
    #
    #     cl = [0, 4]  # wrist up or down
    #     for i in range(0, len(cl)):
    #         c = cl[i]
    #         T_10 = np.linalg.inv(self._AH(1, th, c))
    #         T_16 = T_10 * desired_pos
    #         th[4, c:c + 2] = + \
    #             acos((T_16[2, 3] - self.ur5_d[3]) / self.ur5_d[5])
    #         th[4, c + 2:c + 4] = - \
    #             acos((T_16[2, 3] - self.ur5_d[3]) / self.ur5_d[5])
    #
    #     th = th.real
    #
    #     # **** theta6 ****
    #     # theta6 is not well-defined when sin(theta5) = 0 or when T16(1,3),
    #     # T16(2,3) = 0.
    #     cl = [0, 2, 4, 6]
    #     for i in range(0, len(cl)):
    #         c = cl[i]
    #         T_10 = np.linalg.inv(self._AH(1, th, c))
    #         T_16 = np.linalg.inv(T_10 * desired_pos)
    #         th[5, c:c + 2] = atan2((-T_16[1, 2] / sin(th[4, c])),
    #                                (T_16[0, 2] / sin(th[4, c])))
    #
    #     th = th.real
    #
    #     # **** theta3 ****
    #     cl = [0, 2, 4, 6]
    #     for i in range(0, len(cl)):
    #         c = cl[i]
    #         T_10 = np.linalg.inv(self._AH(1, th, c))
    #         T_65 = self._AH(6, th, c)
    #         T_54 = self._AH(5, th, c)
    #         T_14 = (T_10 * desired_pos) * np.linalg.inv(T_54 * T_65)
    #         P_13 = T_14 * np.matrix([0, -self.ur5_d[3], 0, 1]
    #                                 ).T - np.matrix([0, 0, 0, 1]).T
    #         t3 = cmath.acos((np.linalg.norm(P_13) ** 2 -
    #                          self.ur5_a[1] ** 2 -
    #                          self.ur5_a[2] ** 2) /
    #                         (2 *
    #                          self.ur5_a[1] *
    #                          self.ur5_a[2]))  # norm ?
    #         th[2, c] = t3.real
    #         th[2, c + 1] = -t3.real
    #
    #     # **** theta2 and theta 4 ****
    #     cl = [0, 1, 2, 3, 4, 5, 6, 7]
    #
    #     for i in range(0, len(cl)):
    #         c = cl[i]
    #         T_10 = np.linalg.inv(self._AH(1, th, c))
    #         T_65 = np.linalg.inv(self._AH(6, th, c))
    #         T_54 = np.linalg.inv(self._AH(5, th, c))
    #         T_14 = (T_10 * desired_pos) * T_65 * T_54
    #         P_13 = T_14 * np.matrix([0, -self.ur5_d[3], 0, 1]
    #                                 ).T - np.matrix([0, 0, 0, 1]).T
    #
    #         # theta 2
    #         th[1, c] = -atan2(P_13[1], -P_13[0]) + asin(self.ur5_a[2]
    #                                                     * sin(th[2, c]) / np.linalg.norm(P_13))
    #
    #         # theta 4
    #         T_32 = np.linalg.inv(self._AH(3, th, c))
    #         T_21 = np.linalg.inv(self._AH(2, th, c))
    #         T_34 = T_32 * T_21 * T_14
    #         th[3, c] = atan2(T_34[1, 0], T_34[0, 0])
    #
    #     th = th.real
    #     th = th.T
    #
    #     min_dist = []
    #     for j_p in th:
    #         j_p = np.array(j_p).flatten()
    #         dist_j = np.linalg.norm(j_p - q)
    #         min_dist.append(dist_j)
    #
    #     idx = np.argmin(min_dist)
    #
    #     goal_pose = np.array(th[idx]).flatten()
    #
    #     return goal_pose

    '''
    def solve_IK(self, desired_pos, q):  # T60
        desired_pos2 = m3d.Transform(desired_pos).matrix
#        desired_pos = m3d.Transform(desired_pos).matrix

        th = np.zeros((6, 8))
        P_05 = np.matmul(desired_pos2, np.array([0, 0, -self.ur5_d[5], 1]).transpose()) - np.array([0, 0, 0, 1]).transpose()

        # **** theta1 ****
        psi = atan2(P_05[2 - 1, 0], P_05[1 - 1, 0])
        phi = acos(self.ur5_d[3] / sqrt(P_05[2 - 1, 0] * P_05[2 - 1, 0] + P_05[1 - 1, 0] * P_05[1 - 1, 0]))
        # The two solutions for theta1 correspond to the shoulder
        # being either left or right
        th[0, 0:4] = pi / 2 + psi + phi
        th[0, 4:8] = pi / 2 + psi - phi
        th = th.real

        # **** theta5 ****

        cl = [0, 4]  # wrist up or down
        for i in range(0, len(cl)):
            c = cl[i]
            T_10 = np.linalg.inv(self._AH(1, th, c))
            T_16 = np.matmul(T_10, desired_pos)
            th[4, c:c + 2] = + acos((T_16[2, 3] - self.ur5_d[3]) / self.ur5_d[5])
            th[4, c + 2:c + 4] = - acos((T_16[2, 3] - self.ur5_d[3]) / self.ur5_d[5])

        th = th.real

        # **** theta6 ****
        # theta6 is not well-defined when sin(theta5) = 0 or when T16(1,3), T16(2,3) = 0.
        cl = [0, 2, 4, 6]
        for i in range(0, len(cl)):
            c = cl[i]
            T_10 = np.linalg.inv(self._AH(1, th, c))
            T_16 = np.linalg.inv(T_10 * desired_pos)
            th[5, c:c + 2] = atan2((-T_16[1, 2] / sin(th[4, c])), (T_16[0, 2] / sin(th[4, c])))

        th = th.real

        # **** theta3 ****
        cl = [0, 2, 4, 6]
        for i in range(0, len(cl)):
            c = cl[i]
            T_10 = np.linalg.inv(self._AH(1, th, c))
            T_65 = self._AH(6, th, c)
            T_54 = self._AH(5, th, c)
            T_14 = np.matmul(np.matmul(T_10, desired_pos), np.linalg.inv(np.matmul(T_54, T_65)))
            P_13 = T_14 * np.array([0, -self.ur5_d[3], 0, 1]).transpose() - np.array([0, 0, 0, 1]).transpose()
            t3 = cmath.acos((np.linalg.norm(P_13) ** 2 - self.ur5_a[1] ** 2 - self.ur5_a[2] ** 2) / (2 * self.ur5_a[1] * self.ur5_a[2]))  # norm ?
            th[2, c] = t3.real
            th[2, c + 1] = -t3.real

        # **** theta2 and theta 4 ****
        cl = [0, 1, 2, 3, 4, 5, 6, 7]

        for i in range(0, len(cl)):
            c = cl[i]
            T_10 = np.linalg.inv(self._AH(1, th, c))
            T_65 = np.linalg.inv(self._AH(6, th, c))
            T_54 = np.linalg.inv(self._AH(5, th, c))
            T_14 = np.matmul(np.matmul(np.matmul(T_10, desired_pos), T_65), T_54)
            P_13 = np.matmul(T_14, np.array([0, -self.ur5_d[3], 0, 1]).transpose()) - np.array([0, 0, 0, 1]).transpose()

            # theta 2
            th[1, c] = -atan2(P_13[1], -P_13[0]) + asin(self.ur5_a[2] * sin(th[2, c]) / np.linalg.norm(P_13))

            # theta 4
            T_32 = np.linalg.inv(self._AH(3, th, c))
            T_21 = np.linalg.inv(self._AH(2, th, c))
            T_34 = T_32 * T_21 * T_14
            th[3, c] = atan2(T_34[1, 0], T_34[0, 0])

        th = th.real
        th = th.T

        min_dist = []
        for j_p in th:
            j_p = np.array(j_p).flatten()
            dist_j = np.linalg.norm(j_p - q)
            min_dist.append(dist_j)

        idx = np.argmin(min_dist)

        goal_pose = np.array(th[idx]).flatten()

        return goal_pose
    '''

    def status_chk_old(self):
        # = 로봇 공통 상태 체크
        robotmode = self._program_send("robotmode\n")[0:-1].split(' ')[1]

        # = 로봇 전원이 꺼져있을때 -> 재가동
        if robotmode == 'POWER_OFF':
            self._program_send("power on\n")
            self._program_send("brake release\n")
            time.sleep(5)

        # # safetymode = self._program_send("safetymode\n")[0:-1].split(' ')[1]
        # # = 로봇 상태 체크 # 3.0 버전 이상, 3.11 버전 이하
        # safetymode_old = self._program_send("safetymode\n")[0:-1].split(' ')[1]
        # if safetymode_old != "not":         # : 명령어가 제대로 들어갈 시 not 이 아님
        #     if safetymode_old == "NORMAL":
        #         pass
        #     elif safetymode_old == "PROTECTIVE_STOP":
        #         print("-->>robot : Protective stopped !", file=sys.stderr)
        #         self._program_send("unlock protective stop\n")
        #     elif safetymode_old == "SAFEGUARD_STOP":
        #         print("-->>robot : Safeguard stopped !", file=sys.stderr)
        #         self._program_send("close safety popup\n")
        #     elif safetymode_old == "FAULT":                                 # : 추가
        #         print("-->>robot : Safety fault !", file=sys.stderr)
        #         self._program_send("restart safety\n")
        #     else:
        #         print("-->>robot : Unreachable position self.obj_pos")
        #         print(safetymode_old)

        # = 로봇 상태 체크 # 3.11 버전 이상
        safetymode_new = self._program_send("safetystatus\n")[0:-1].split(' ')[1]

        if safetymode_new != "not":  # : 명령어가 제대로 들어갈 시 not 이 아님
            if safetymode_new == "NORMAL":
                pass
            elif safetymode_new == "PROTECTIVE_STOP":
                print("-->>robot : Protective stopped !", file=sys.stderr)
                self._program_send("unlock protective stop\n")
            elif safetymode_new == "SAFEGUARD_STOP":
                print("-->>robot : Safeguard stopped !", file=sys.stderr)
                self._program_send("close safety popup\n")
            elif safetymode_new == "FAULT" or safetymode_new == "VIOLATION":                                 # : 추가
                print("-->>robot : System fault !", file=sys.stderr)
                self._program_send("restart safety\n")
            else:
                print("-->>robot : Unreachable position self.obj_pos")
                print(safetymode_new)
            self._program_send("close safety popup\n")

    def status_chk(self):
        # ~status chk reward, cur_angle, next_angle use ?
        count1 = 0
        while count1 < 10:
            # = 팝업 메세지들을 제거함
            self._program_send("close safety popup\n")
            self._program_send("close popup\n")

            # = 로봇의 상태를 체크
            robotmode = self._program_send("robotmode\n")[0:-1].split(' ')[1]

            if robotmode == 'RUNNING':
                pass
            elif robotmode == 'POWER_ON':
                time.sleep(1)
                continue
            elif robotmode == 'POWER_OFF':
                self._program_send("unlock protective stop\n")
                self._program_send("restart safety\n")
                time.sleep(2)
                self._program_send("power on\n")
                time.sleep(2)
                self._program_send("brake release\n")
                time.sleep(2)
                self._program_send("close popup\n")
                time.sleep(2)
                continue
            elif robotmode == 'BOOTING' or robotmode == 'IDLE':
                time.sleep(5)
                continue
            else:   # : 20200207 'NO_CONTROLLER'
                print("-->>robot : {} robotmode fail".format(count1))
                print(robotmode + " ... 정의되지 않은 오류")     # : 20200121 디버깅용 ##
                continue

            # robotmode = self.status_robotmode()
            # safetymode = self._program_send("safetymode\n")[0:-1].split(' ')[1]   # : 20200117 수정##

            # = 로봇의 안정성 검사
            safetymode = self._program_send("safetystatus\n")[0:-1].split(' ')[1]

            if safetymode != "not":  # : 명령어가 제대로 들어갈 시 not 이 아님
                if safetymode == "NORMAL" and robotmode == 'RUNNING':
                    break
                elif safetymode == "PROTECTIVE_STOP":
                    print("-->>robot : {} Protective stopped !".format(count1), file=sys.stderr)
                    self._program_send("unlock protective stop\n")
                    self._program_send("close safety popup\n")
                    time.sleep(2)
                elif safetymode == "SAFEGUARD_STOP":
                    print("-->>robot : {} Safeguard stopped !".format(count1), file=sys.stderr)
                    self._program_send("unlock protective stop\n")
                    self._program_send("close safety popup\n")
                    time.sleep(2)
                elif safetymode == "FAULT" or safetymode == "VIOLATION":
                    print("-->>robot : {} System fault !".format(count1), file=sys.stderr)
                    self._program_send("unlock protective stop\n")
                    self._program_send("restart safety\n")
                    self._program_send("close safety popup\n")
                    self._program_send("close popup\n")
                    time.sleep(2)
                    continue
                else:
                    print("-->>robot : {} Unreachable position self.obj_pos".format(count1))
                    print(safetymode + " ... 정의되지 않은 오류")    # : 20200121 디버깅용 ##    IDLE 처리 필요
                    self._program_send("close safety popup\n")
                    time.sleep(5)

            # time.sleep(1)
            count1 += 1

    # # = 현재 포즈를 넣으면, 현재 포즈 반영하여 손목 각도 유지
    # def action_mod(self, pose, now_action_joint):  # : action 6개 x, y, z, rz, 90, 0   / now_action_j 현재 joint
    #     pose_mod = [pose[1], -pose[0], pose[2], pose[3], pose[4], pose[5]]  # : 3축 y, -x, z / rz, 90, 0
    #     action_jointrotate = now_action_joint.copy()
    #     action_jointrotate[5] += pose[3]                    # : 현재 손목(rx) 각도를 현재 joint로 계산
    #     action_pose = self.solve_FK(action_jointrotate)     # : FK를 이용해 rx ry rz 계산
    #     pose_mod[3:6] = action_pose[3:6]                    # : 계산된 r을 action에 입력
    #     return pose_mod
    #
    # def action_mod_90(self, pose, now_action_joint):  # : action 6개 x, y, z, rz, 90, 0   / now_action_j 현재 joint
    #     pose_mod = [pose[1], -pose[0], pose[2], pose[3], pose[4], pose[5]]  # : 3축 y, -x, z / rz, 90, 0
    #     action_jointrotate = now_action_joint.copy()
    #     action_jointrotate[5] = action_jointrotate[0]       # : 현재 손목(rx) 각도를 현재 joint로 계산
    #     action_pose = self.solve_FK(action_jointrotate)     # : FK를 이용해 rx ry rz 계산
    #     pose_mod[3:6] = action_pose[3:6]                    # : 계산된 r을 action에 입력
    #     return pose_mod
    #
    # def action_mod_270(self, pose, now_action_joint):  # : action 6개 x, y, z, rz, 90, 0   / now_action_j 현재 joint
    #     pose_mod = [pose[1], -pose[0], pose[2], pose[3], pose[4], pose[5]]  # : 3축 y, -x, z / rz, 90, 0
    #     action_jointrotate = now_action_joint.copy()
    #     action_jointrotate[5] = action_jointrotate[0]-(180*(math.pi/180))       # : 현재 손목(rx) 각도를 현재 joint로 계산
    #     action_pose = self.solve_FK(action_jointrotate)     # : FK를 이용해 rx ry rz 계산
    #     pose_mod[3:6] = action_pose[3:6]                    # : 계산된 r을 action에 입력
    #     return pose_mod
    #
    # def action_mod2(self, action, now_action_j):
    #     action_mod = [action[1], -action[0], action[2], action[3], action[4], action[5]]  # : 3축 x, y, z / rz, 90, 0
    #     now_action_jr = copy.deepcopy(now_action_j)
    #     now_action_jr[5] += action[3]  # : 현재 손목(rx) 각도를 현재 joint로 계산   ##
    #     action_rotation = self.solve_FK(now_action_jr)  # : FK를 이용해 rx ry rz 계산
    #     action_mod[3:6] = action_rotation[3:6]          # : 계산된 r을 action에 입력
    #     return action_mod
    #
    # def action_mod3(self, action, now_action_j):
    #     action_mod = [action[1], -action[0], action[2], action[3], action[4], action[5]]  # : 3축 x, y, z / rz, 90, 0
    #     now_action_jr = now_action_j.copy()
    #     now_action_jr[5] += action[3]  # : 현재 손목(rx) 각도를 현재 joint로 계산
    #     q = self.getj()
    #     action_rotation = self.solve_IK(action_mod, q)  # : FK를 이용해 rx ry rz 계산
    #     action_mod = action_rotation  # : 계산된 r을 action에 입력
    #     return action_mod

    def movej(self, j, acc=0.1, vel=0.1, wait=True):
        is_Exception = False
        try:
            self.rob.movej(j, acc, vel, wait)
            # self.status_chk()
        except urx.RobotException:
            self.status_chk()
            is_Exception = True

            temp_ = self.rob.getj()
            self.rob.movej(temp_, acc, vel, wait)

        return is_Exception

    def movel(self, p, acc=0.1, vel=0.1, wait=True):
        is_Exception = False
        try:
            self.rob.movel(p, acc, vel, wait)
            # self.status_chk()
        except urx.RobotException:
            self.status_chk()
            is_Exception = True

            temp_ = self.rob.getl()
            self.rob.movel(temp_, acc, vel, wait)

        return is_Exception

    def movels(self, p, acc=0.1, vel=0.1, radius=0.01, wait=True):
        is_Exception = False
        try:
            self.rob.movels(p, acc, vel, radius=radius, wait=wait)
            # self.status_chk()
        except urx.RobotException:
            self.status_chk()
            is_Exception = True

        return is_Exception

    def movec(self, ps, pe, acc=0.1, vel=0.1, wait=True):
        is_Exception = False
        try:
            self.rob.movec(ps, pe, acc, vel, wait)
            # self.status_chk()
        except urx.RobotException:
            self.status_chk()
            is_Exception = True

        return is_Exception

    def getj(self):
        return self.rob.getj()

    def getl(self):
        return self.rob.getl()

    def set_tcp(self, tcp):
        return self.rob.set_tcp(tcp)

    def _wait_for_move(self, target, threshold=None, timeout=5, joints=False, force_thr=False):

        is_Exception = False
        try:
            self.rob._wait_for_move(target, threshold, timeout, joints, force_thr)

        except urx.RobotException:
            self.status_chk()
            is_Exception = True

        except urx.ForceException:
            self.status_chk()
            is_Exception = True

        # return is_Exception

    def send_program(self, prog):
        self.rob.send_program(prog)

