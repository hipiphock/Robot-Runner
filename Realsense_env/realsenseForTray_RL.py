import pyrealsense2 as rs
from Robot_env.config import M_k2b
import numpy as np
import cv2
import sys
import time
import ctypes
import logging

logger = logging.getLogger("realsense")
# handler = logging.StreamHandler(sys.stderr)
# handler.setLevel(logging.INFO)
# logger.addHandler(handler)

class Realsense:
    def __init__(self):
        logger.info("initializing Realsense")
        while True:
            try:
                self.file_num = 0
                self.pipeline = None
                self.config = None
                self.frames = None
                self.color_frame = None
                self.align = None
                self.frameset = None
                self.aligned_depth_frame = None
                self.depth_intrin = None
                self.color_intrin = None
                self.depth_to_color_extrin = None
                self.initialize()
                self.raw_color = None
                break
            except:
                logger.error("retrying Realsense connect")
        logger.info("Realsense Ready")

    def initialize(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(self.config)
        self.frames = self.pipeline.wait_for_frames()

        # Camera burn in.
        for _ in range(30):
            self.frames = self.pipeline.wait_for_frames()
            _ = self.frames.get_color_frame()

        self.color_frame = self.frames.get_color_frame()

        self.align = rs.align(rs.stream.color)
        self.frameset = self.align.process(self.frames)

        self.aligned_depth_frame = self.frameset.get_depth_frame()
        self.depth_intrin = self.aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        self.color_intrin = self.color_frame.profile.as_video_stream_profile().intrinsics
        self.depth_to_color_extrin = self.aligned_depth_frame.profile.get_extrinsics_to(self.color_frame.profile)

        depth_image = np.asanyarray(self.aligned_depth_frame.get_data())
        color_image = np.asanyarray(self.color_frame.get_data())

    def capture(self):
        self.frames = self.pipeline.wait_for_frames()
        self.color_frame = self.frames.get_color_frame()
        self.raw_color = np.asanyarray(self.color_frame.get_data())  # RGB

        # : 20200115 ... 임시 ########################
        self.frames = self.pipeline.wait_for_frames()

        self.align = rs.align(rs.stream.color)
        self.frameset = self.align.process(self.frames)
        self.depth_frame = self.frameset.get_depth_frame()
        # depth_frame = frames.get_depth_frame()

        self.depth_intrin = self.depth_frame.profile.as_video_stream_profile().intrinsics

        return self.raw_color

    def pxl2xyz(self, obj_pos, mean_xy_point):
        # : 20200115 ... 임시 ########################
        # self.frames = self.pipeline.wait_for_frames()
        #
        # self.align = rs.align(rs.stream.color)
        # self.frameset = self.align.process(self.frames)
        # self.depth_frame = self.frameset.get_depth_frame()
        # # depth_frame = frames.get_depth_frame()
        #
        # self.depth_intrin = self.depth_frame.profile.as_video_stream_profile().intrinsics

        if obj_pos.__len__() is 0:                     ## ! 죽는 지점 임시
            return None

        xyz_list = []
        # ---- ---- ---- ----
        c_y = int(mean_xy_point[0])
        c_x = int(mean_xy_point[1])

        camera_x = []
        camera_y = []
        camera_z = []
        for i in range(c_y - 3, c_y + 3 + 1):
            for j in range(c_x - 3, c_x + 3 + 1):
                depth_1 = self.depth_frame.get_distance(j, i)
                if depth_1 == 0:
                    continue
                depth_point_1 = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [j, i], depth_1)
                camera_y.append(depth_point_1[1])
                camera_x.append(depth_point_1[0])
                camera_z.append(depth_point_1[2])

        if 0 < camera_y.__len__() and 0 < camera_x.__len__() and 0 < camera_z.__len__():
            camera_y_avg1 = sum(camera_y) / camera_y.__len__()
            camera_x_avg1 = sum(camera_x) / camera_x.__len__()
            camera_z_avg1 = sum(camera_z) / camera_z.__len__()
        else:
            aaa = 0     # -#- 디버깅용

        camera_x2 = []
        camera_y2 = []
        camera_z2 = []
        for k in range(c_y - 3, c_y + 3 + 1):
            for l in range(c_x - 3, c_x + 3 + 1):
                depth_2 = self.depth_frame.get_distance(l, k)
                if depth_2 == 0:
                    continue
                depth_point_2 = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [l, k], depth_2)
                if ((camera_y_avg1 - abs(camera_y_avg1 / 10)) < depth_point_2[1] < (
                        camera_y_avg1 + abs(camera_y_avg1 / 10))) and \
                        ((camera_x_avg1 - abs(camera_x_avg1 / 10)) < depth_point_2[0] < (
                                camera_x_avg1 + abs(camera_x_avg1 / 10))) and \
                        ((camera_z_avg1 - abs(camera_z_avg1 / 10)) < depth_point_2[2] < (
                                camera_z_avg1 + abs(camera_z_avg1 / 10))):
                    camera_y2.append(depth_point_2[1])
                    camera_x2.append(depth_point_2[0])
                    camera_z2.append(depth_point_2[2])
        if camera_y.__len__() > 0 and camera_x.__len__() > 0 and camera_z.__len__() > 0:
            camera_y_avg2 = sum(camera_y2) / camera_y2.__len__()    # -#- 오류
            camera_x_avg2 = sum(camera_x2) / camera_x2.__len__()
            camera_z_avg2 = sum(camera_z2) / camera_z2.__len__()

            cam_pos = np.array([[camera_x_avg2, camera_y_avg2, camera_z_avg2, 1]])  # -#-
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

    def path2xyz(self, path):

        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

        pxl_patch = path

        # Seg 이미지에서의 경로를 카메라 기준 경로로 바꿔주는 작업 - 수정 필요
        for idx, [y, x] in enumerate(pxl_patch):
            pxl_patch[idx][1] = x * (916/360) + 187
            pxl_patch[idx][0] = y * (916/360) - (916-720)/2

        pxl_patch = np.array(pxl_patch).astype(np.uint32)  # round? int?

        xyz_list = []
        camera_x = []
        camera_y = []
        camera_z = []
        for c_y, c_x in pxl_patch:
            depth = depth_frame.get_distance(c_x, c_y)
            if depth == 0.0:
                continue
            depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [c_x, c_y], depth)
            camera_y.append(depth_point[0])
            camera_x.append(depth_point[1])
            camera_z.append(depth_point[2])

        camera_y_avg = sum(camera_y) / camera_y.__len__()
        camera_x_avg = sum(camera_x) / camera_x.__len__()
        camera_z_avg = sum(camera_z) / camera_z.__len__()

        cam_pos = np.array([[camera_y_avg, camera_x_avg, camera_z_avg, 1]])
        xyz = np.dot(M_k2b, cam_pos.T)
        xyz_list.append(xyz)

        xyz_list = np.array(xyz_list)

        while np.any(np.isinf(xyz_list)) or np.any(np.isnan(xyz_list)):
            nan_idx = np.sort(np.transpose(np.argwhere(np.isnan(xyz_list))[0::3])[0])
            for x in reversed(nan_idx):
                xyz_list = np.delete(xyz_list, x, 0)

        if np.any(np.isinf(xyz_list)) or np.any(np.isnan(xyz_list)):
            return None
        else:
            return xyz_list
