"""Scripted policy 생성 클래스"""
import copy
import random
import numpy as np
import math

import cv2

#
# class camera_realtimeXYZ:
#     # camera variables
#     cam_mtx = None
#     dist = None
#     newcam_mtx = None
#     roi = None
#     rvec1 = None
#     tvec1 = None
#     R_mtx = None
#     Rt = None
#     P_mtx = None
#
#     # images
#     img = None
#
#     def __init__(self):
#
#         imgdir = "/home/pi/Desktop/Captures/"
#         savedir = "camera_data/"
#
#         self.cam_mtx = np.load(savedir + 'cam_mtx.npy')
#         self.dist = [0.0, 0.0, 0.0, 0.0, 0.0]
#         self.newcam_mtx = np.load(savedir + 'newcam_mtx.npy')
#         self.roi = np.load(savedir + 'roi.npy')
#         self.rvec1 = np.load(savedir + 'rvec1.npy')
#         self.tvec1 = np.load(savedir + 'tvec1.npy')
#         self.R_mtx = np.load(savedir + 'R_mtx.npy')
#         self.Rt = np.load(savedir + 'Rt.npy')
#         self.P_mtx = np.load(savedir + 'P_mtx.npy')
#
#         s_arr = np.load(savedir + 's_arr.npy')
#         self.scalingfactor = s_arr[0]
#
#         self.inverse_newcam_mtx = np.linalg.inv(self.newcam_mtx)
#         self.inverse_R_mtx = np.linalg.inv(self.R_mtx)
#
#     def undistort_image(self, image):
#         image_undst = cv2.undistort(image, self.cam_mtx, self.dist, None, self.newcam_mtx)
#
#         return image_undst
#
#     def load_background(self, background):
#         self.bg_undst = self.undistort_image(background)
#
#     def detect_xyz(self, image):
#
#         image_src = image.copy()
#         img = image_src
#
#         XYZ = []
#
#         cx = detected_points[i][4]
#         cy = detected_points[i][5]
#
#         XYZ.append(self.calculate_XYZ(cx, cy))
#
#         return img, XYZ
#
#     def calculate_XYZ(self, u, v):
#
#         # Solve: From Image Pixels, find World Points
#         uv_1 = np.array([[u, v, 1]], dtype=np.float32)
#         uv_1 = uv_1.T
#         suv_1 = self.scalingfactor * uv_1
#         xyz_c = self.inverse_newcam_mtx.dot(suv_1)
#         xyz_c = xyz_c - self.tvec1
#         XYZ = self.inverse_R_mtx.dot(xyz_c)
#
#         return XYZ
#
#     def truncate(self, n, decimals=0):
#         n = float(n)
#         multiplier = 10 ** decimals
#         return int(n * multiplier) / multiplier


class scriptedPolicy:

    APPROACH_STEP = 3

    def __init__(self, tray_z=None):  # : scripted policy를 environment에 맞춰서 초기화
        # using bellow parameter creating scripted policy
        self.X_RANGE = 0.30  # 30cm
        self.Y_RANGE = 0.40
        self.Z_VALUE = 0.17

        # Sample boundary
        self.X = [0.244, 0.684]
        self.Y = [-0.285, 0.275]
        self.Z = [0.045, 0.245]

        self.tray_z = tray_z       ## pyb 하드코딩 할 것

        #step policy variable
        self._stepcount = 0
        self._initpos = None
        self._initxyzrobopos = None

    # def getTruePolicy(self, rob, cam, obj_pos, obj_id):
    #
    #     self._initxyzrobopos = self.InitialPos(cam, obj_pos)
    #
    #     action = list(self._initxyzrobopos)
    #
    #     return action

    def getScriptedPolicy(self, rob, obj_pos, obj_id, depth_img, depth_int):
        action = []
        action_ext = []
        scriptedEnd = False
        eps = 0.00001

        if self._stepcount == 0:        # : 스텝의 처음 시작시
            # 1. Move UR5 arms over an object.
            self._initxyzrobopos = self.InitialPosSampling(obj_pos, depth_img, depth_int)

            xpos = 0.52
            ypos = -0.18

            self.rot = math.pi * (np.random.uniform(low=0.01, high=0.99) - 0.5)
            # self.rot = math.pi * 0.25
            self._initxyzrobopos = self._initxyzrobopos + [0.0, -math.pi, 0.0]
            action = list(self._initxyzrobopos)
            action[0] = xpos
            action[1] = ypos
            gripper_open = (np.random.uniform() / 2.0) + 0.5
            gripper_close = 1.0 - gripper_open
            terminal = (np.random.uniform() / 2.0) - eps
            action_ext = [gripper_close, gripper_open, terminal]

        elif 0 < self._stepcount <= self.APPROACH_STEP:     # : 스텝이 진행 중이면서 다가가기 중일때

            # 3. z approach ( N step )
            if self._stepcount == 1:
                approach_action = self._initxyzrobopos.copy()
                approach_action[3] = self.rot
                approachDepth = approach_action[2] - self.tray_z  # pyb

                if obj_id == 8:
                    approachDepth -= 0.02
                if obj_id == 14 or obj_id == 24:
                    approachDepth += 0.01

                rand_num = np.random.randint(0, 10, 1)
                if rand_num[0] < 2:
                    approachNoise = np.random.normal() / 100.0
                else:
                    approachNoise = 0.0

                approachDepth -= approachNoise

                if approachDepth > self.Z[1]:
                    approachDepth = self.Z[1]

                approach_ratio = self.SeperateRandomly(approachDepth, self.APPROACH_STEP)  # 어떤 기능을 하는 함수?
                approach_ratio = approach_ratio.tolist()
                self.approach_pos = []

                approach_ratio_sum = 0.0
                for i in range(self.APPROACH_STEP):
                    approach_action_temp = approach_action.copy()
                    approach_ratio_sum += approach_ratio[i]
                    if i > 0:
                        approach_action_temp[2] -= approach_ratio_sum
                    self.approach_pos.append(approach_action_temp)

            # approach_ratio를 보고 정해진 높이 만큼 내려왔으면 self._stepcount = self.APPROACH_STEP + 1 로 해준다  pyb
            action = self.approach_pos[self._stepcount - 1]
            gripper_open = (np.random.uniform() / 2.0) + 0.5
            gripper_close = 1.0 - gripper_open
            terminal = (np.random.uniform() / 2.0) - eps
            action_ext = [gripper_close, gripper_open, terminal]
        elif self._stepcount == self.APPROACH_STEP + 1:                         # : 다가가기 스텝이 끝난 직후
            effstate = rob.getl() # 4. grasp  pyb : 왜 6 X 3 차원인가?
            effstate = [-effstate[1], effstate[0], effstate[2], self.rot, -math.pi, 0.0]

#            action = list(effstate[0]) + list(self._env._p.getEulerFromQuaternion(effstate[1])) + [1]
            action = effstate

            gripper_open = (np.random.uniform() / 2.0) - eps
            gripper_close = 1.0 - gripper_open
            terminal = (np.random.uniform() / 2.0) - eps
            action_ext = [gripper_close, gripper_open, terminal]

        elif self.APPROACH_STEP + 2 <= self._stepcount <= self.APPROACH_STEP * 2 + 1:   # : 다가가기 스텝이 끝난뒤
            # 들어올리기
            if self._stepcount == self.APPROACH_STEP + 2:
                effstate = rob.getl()
                effstate = [-effstate[1], effstate[0], effstate[2], self.rot, -math.pi, 0.0]
                approachDepth = self._initxyzrobopos[2] - effstate[2]   # 수정 필요
                approach_ratio = self.SeperateRandomly(approachDepth, self.APPROACH_STEP)
                approach_ratio = approach_ratio.tolist()
                self.approach_pos = []
#                approach_action = list(effstate[0]) + list(self._env._p.getEulerFromQuaternion(effstate[1])) + [1]
                approach_action = effstate
                approach_ratio_sum = 0.0
                for i in range(self.APPROACH_STEP):
                    approach_action_temp = approach_action.copy()
                    approach_ratio_sum += approach_ratio[i]
                    approach_action_temp[2] += approach_ratio_sum
                    self.approach_pos.append(approach_action_temp)

            action = self.approach_pos[self._stepcount - self.APPROACH_STEP - 2]
            gripper_open = (np.random.uniform() / 2.0) - eps
            gripper_close = 1.0 - gripper_open
            terminal = (np.random.uniform() / 2.0) - eps
            action_ext = [gripper_close, gripper_open, terminal]

            if self._stepcount == self.APPROACH_STEP * 2 + 1:
                gripper_open = (np.random.uniform() / 2.0) - eps
                gripper_close = 1.0 - gripper_open
                terminal = (np.random.uniform() / 2.0) - eps
                action_ext = [gripper_close, gripper_open, terminal]
                scriptedEnd = True

        if scriptedEnd:             # : scriptedPolicy 가 끝나면
            self._stepcount = 0     # : step flag 설정
        else:
            self._stepcount += 1    # : scriptedPolicy 진행중 step flag count
        return action, action_ext, scriptedEnd
    #
    # def InitialPos(self, cam, obj_pos):
    #     xyz_pos = cam.pxl2xyz(obj_pos)
    #
    #     if xyz_pos is not None:
    #         xyz_pos = [-xyz_pos[1], xyz_pos[0] - 0.03, xyz_pos[2]]
    #         noisex = 0.0
    #         noisey = 0.0
    #
    #         xpos = xyz_pos[0]
    #         xpos += noisex
    #         if xpos > self.X[1]:
    #             xpos = self.X[1]
    #         if xpos < self.X[0]:
    #             xpos = self.X[0]
    #         ypos = xyz_pos[1]
    #         ypos += noisey
    #         if ypos > self.Y[1]:
    #             ypos = self.Y[1]
    #         if ypos < self.Y[0]:
    #             ypos = self.Y[0]
    #     else:
    #         xpos = np.random.uniform(self.X[0], self.X[1], 1)
    #         ypos = np.random.uniform(self.Y[0], self.Y[1], 1)
    #
    #     zpos = self.tray_z
    #     sample_pos = [xpos, ypos, zpos]
    #
    #     return sample_pos

    # def pxl2xyz(self, obj_pos):
    #
    #     frames = self.pipeline.wait_for_frames()
    #
    #     #        align = rs.align(rs.stream.color)
    #     #        frameset = align.process(frames)
    #     #        depth_frame = frameset.get_depth_frame()
    #     depth_frame = frames.get_depth_frame()
    #
    #     depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    #
    #     pxl_patch = []
    #     if obj_pos is None:  ## ! 죽는 지점 임시
    #         return None
    #     start_pxl = obj_pos - np.array([2, 2])  ## ! 죽는 지점 발생 - 'NoneType' and 'int'
    #
    #     for i in range(5):
    #         for j in range(5):
    #             pxl_patch.append(start_pxl + np.array([i, j]))
    #
    #     for idx, [y, x] in enumerate(pxl_patch):
    #         pxl_patch[idx][1] = x * (916 / 360) + 187
    #         pxl_patch[idx][0] = y * (916 / 360) - (916 - 720) / 2
    #
    #     pxl_patch = np.array(pxl_patch).astype(np.uint32)  # round? int?
    #
    #     xyz_list = []
    #
    #     camera_x = []
    #     camera_y = []
    #     camera_z = []
    #     for c_y, c_x in pxl_patch:
    #         depth = depth_frame.get_distance(c_x, c_y)
    #         if depth == 0.0:
    #             continue
    #         if depth == None:
    #             continue
    #
    #         depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [c_x, c_y], depth)
    #         camera_y.append(depth_point[0])
    #         camera_x.append(depth_point[1])
    #         camera_z.append(depth_point[2])
    #
    #     if camera_y.__len__() > 0 and camera_x.__len__() > 0 and camera_z.__len__() > 0:
    #         camera_y_avg = sum(camera_y) / camera_y.__len__()
    #         camera_x_avg = sum(camera_x) / camera_x.__len__()
    #         camera_z_avg = sum(camera_z) / camera_z.__len__()
    #
    #         cam_pos = np.array([[camera_y_avg, camera_x_avg, camera_z_avg, 1]])
    #         xyz = np.dot(M_k2b, cam_pos.T)
    #         xyz = xyz.flatten()[:-1]
    #         xyz_list.append(xyz)
    #
    #         xyz_list = np.array(xyz_list)
    #
    #         while np.any(np.isinf(xyz_list)) or np.any(np.isnan(xyz_list)):
    #             nan_idx = np.sort(np.transpose(np.argwhere(np.isnan(xyz_list))[0::3])[0])
    #             for x in reversed(nan_idx):
    #                 xyz_list = np.delete(xyz_list, x, 0)
    #
    #         mean_xyz = np.mean(xyz_list, axis=0)
    #     else:
    #         return None
    #
    #     if np.any(np.isinf(mean_xyz)) or np.any(np.isnan(mean_xyz)):
    #         return None
    #     else:
    #         return mean_xyz

    ####
    def pxl2xyz(self, obj_pos, depth_img, depth_int):
        import tkinter
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        # plt.imshow(depth_img)
        plt.imsave("./depth_img.png", depth_img)
        # plt.show()

        # Error: 0.00333206
        # 0.00387113, 0.698978, -0.732627, 0.611511,
        # 1.01034, -0.00563126, -0.00890769, -0.463882,
        # -0.00929498, -0.741844, -0.66469, 0.449643,
        # 0, 0, 0, 1,
        # Average
        # Error: % f
        # mm
        # 0.00333206
        # Inlier
        # Count: % d
        # 376

        # M_k2b = np.array([[0.0932382, 0.791103, -0.295438, 0.293495],
        #                   [0.603681, 0.0593633, 0.0369499, -0.491357],
        #                   [-0.000116229, -0.172577, -0.295063, 0.179793],
        #                   [0, 0, 0, 1]])
        M_k2b = np.array([[0.00387113, 0.698978, -0.732627, 0.611511],
                          [1.01034, -0.00563126, -0.00890769, -0.463882],
                          [-0.00929498, -0.741844, -0.66469, 0.449643],
                          [0, 0, 0, 1]])
        depth_frame = copy.deepcopy(depth_img)
        depth_intrin = copy.deepcopy(depth_int)

        pxl_patch = []
        if obj_pos is None:  ## ! 죽는 지점 임시
            return None
        start_pxl = obj_pos - np.array([2, 2])  ## ! 죽는 지점 발생 - 'NoneType' and 'int'

        for i in range(5):
            for j in range(5):
                pxl_patch.append(start_pxl + np.array([i, j]))

        for idx, [y, x] in enumerate(pxl_patch):
            pxl_patch[idx][1] = x * (916 / 360) + 187
            pxl_patch[idx][0] = y * (916 / 360) - (916 - 720) / 2

        pxl_patch = np.array(pxl_patch).astype(np.uint32)  # round? int?

        xyz_list = []

        camera_x = []
        camera_y = []
        camera_z = []

        height = depth_intrin[1]
        width = depth_intrin[0]
        ppx = depth_intrin[2]
        ppy = depth_intrin[3]
        fx = depth_intrin[4]
        fy = depth_intrin[5]
        model = depth_intrin[6]
        coeffs = depth_intrin[7]

        for c_y, c_x in pxl_patch:
            # depth = depth_frame[c_x, c_y]
            depth = depth_frame[c_y, c_x]#/1000.0
            if depth == 0.0:
                continue
            if depth is None:
                continue
            ################################ #++# 20191217
            # : RS2_DISTORTION_INVERSE_BROWN_CONRADY)
            x = (c_x - ppx) / fx
            y = (c_y - ppy) / fy

            # r2 = x * x + y * y
            # f = 1 + coeffs[0] * r2 + coeffs[1] * r2 * r2 + coeffs[4] * r2 * r2 * r2
            #
            # ux = x * f + 2 * coeffs[2] * x * y + coeffs[3] * (r2 + 2 * x * x)
            # uy = y * f + 2 * coeffs[3] * x * y + coeffs[2] * (r2 + 2 * y * y)
            # x = ux
            # y = uy

            # depth_point[1] = depth * x
            # depth_point[0] = depth * y
            # depth_point[2] = depth
            depth_point = [depth * x, depth * y,  depth]
            ################################
            # depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [c_x, c_y], depth)
            camera_x.append(depth_point[0])
            camera_y.append(depth_point[1])
            camera_z.append(depth_point[2])

        if 0 < camera_y.__len__() and 0 < camera_x.__len__() and 0 < camera_z.__len__():
            camera_x_avg = sum(camera_x) / camera_x.__len__()
            camera_y_avg = sum(camera_y) / camera_y.__len__()
            camera_z_avg = sum(camera_z) / camera_z.__len__()

            cam_pos = np.array([[camera_x_avg, camera_y_avg, camera_z_avg, 1]])
            xyz = np.dot(M_k2b, cam_pos.T)
            xyz = xyz.flatten()[:-1]
            xyz_list.append(xyz)

            xyz_list = np.array(xyz_list)

            while np.any(np.isinf(xyz_list)) or np.any(np.isnan(xyz_list)):
                nan_idx = np.sort(np.transpose(np.argwhere(np.isnan(xyz_list))[0::3])[0])
                for x in reversed(nan_idx):
                    xyz_list = np.delete(xyz_list, x, 0)

            mean_xyz = np.mean(xyz_list, axis=0)
        else:
            return None

        if np.any(np.isinf(mean_xyz)) or np.any(np.isnan(mean_xyz)):
            return None
        else:
            return mean_xyz

    def InitialPosSampling(self, obj_pos, depth_img, depth_int):
        xyz_pos = self.pxl2xyz(obj_pos, depth_img, depth_int)

        if xyz_pos is not None:
            # xyz_pos = [-xyz_pos[1], xyz_pos[0] - 0.03, xyz_pos[2]]
            xyz_pos_ = [-xyz_pos[1], xyz_pos[0], xyz_pos[2]]
            noisex = (random.random() - 0.5) * 0.01
            noisey = (random.random() - 0.5) * 0.01

            xpos = xyz_pos_[0]
            xpos += noisex
            if self.X[1] < xpos:
                xpos = self.X[1]
            if xpos < self.X[0]:
                xpos = self.X[0]
            ypos = xyz_pos_[1]
            ypos += noisey
            if self.Y[1] < ypos:
                ypos = self.Y[1]
            if ypos < self.Y[0]:
                ypos = self.Y[0]
        else:
            xpos = np.random.uniform(self.X[0], self.X[1], 1)
            ypos = np.random.uniform(self.Y[0], self.Y[1], 1)

        zpos = self.tray_z + self.Z_VALUE
        sample_pos = [xpos, ypos, zpos]

        return sample_pos

    def SeperateRandomly(self, value, N):
        score_sum = 0
        scorelist = []
        for i in range(N):
            rand_val = random.random()
            scorelist.append(rand_val)
            score_sum += rand_val

        scorelist = np.array(scorelist)
        scorelist /= score_sum
        ratio = scorelist * value

        return ratio

    def reset(self):
        self._stepcount = 0

