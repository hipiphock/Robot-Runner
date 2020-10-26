import tensorflow as tf
import numpy as np
import math
import datetime, os, yaml
import cv2
from Robot_env.config import RL_Obj_List
from PIL import Image as Im
import sys
import time


class SegmentationGraph():
	"""  Importing and running isolated TF graph """

	def __init__(self, loc):
		# Create local graph and use it in the session

		self.graph = tf.Graph()
		self.sess = tf.Session(graph=self.graph)

		with self.graph.as_default():
			# Import saved model from location 'loc' into local graph
			ckpt_info_file = loc + "checkpoint"

			os.path.join(loc, "check")

			info = yaml.load(open(ckpt_info_file, "r"))
			assert 'model_checkpoint_path' in info
			#      most_recent_ckpt = "%s\\%s" % (self.ckpt_dir, info['model_checkpoint_path'])
			most_recent_ckpt = "%s" % (info['model_checkpoint_path']) + '.meta'
			saver = tf.train.import_meta_graph(most_recent_ckpt, clear_devices=True)
			most_recent_ckpt = "%s" % (info['model_checkpoint_path'])
			saver.restore(self.sess, most_recent_ckpt)
			# Get activation function from saved collection
			# You may need to change this in case you name it differently
			# self.segmentation = tf.get_collection('argmax_output')[0]
			self.image = tf.get_collection('image')
			self.segmentation = tf.get_collection('segmented_image')
			self.new_segmented_array = np.zeros([1280, 720])

	def run(self, image):
		# The 'x' corresponds to name of input placeholder
		# result = self.sess.run(self.segmentation, feed_dict={'input:0': [image]})[0]
		img_name = image.copy()

		cv2.imshow('show_img', img_name)
		cv2.waitKey(1)

		input_img = img_name.astype(np.float32)

		# result = self.sess.run(self.segmentation, feed_dict={'image': [resize_img]})[0]
		# new_segmented_array, new_segmented_color_array = self.new_segmented_image_by_connected_component(result)  # : new_segmented_image_by_connected_component 이미지 처리/ 노이즈제거
		# self.new_segmented_array = new_segmented_array

		result = self.sess.run(self.segmentation, feed_dict={'image:0': [input_img]})  # : 1280 720
		new_segmented_array, new_segmented_color_array = self.new_segmented_image_by_connected_component(result)
		self.new_segmented_array = new_segmented_array

		return new_segmented_array, new_segmented_color_array

	# ##-- 안쓰고있음
	def read_filelist(self, img_path, seg_path):

		imglist = []
		seglist = []

		for (path, dir, files) in os.walk(img_path):
			for filename in files:
				ext = os.path.splitext(filename)[-1]
				if ext == '.png':
					full_filename = os.path.join(path, filename)
					imglist.append(filename)

		for (path, dir, files) in os.walk(seg_path):
			for filename in files:
				ext = os.path.splitext(filename)[-1]
				if ext == '.png':
					full_filename = os.path.join(path, filename)
					seglist.append(filename)

		return imglist, len(imglist), seglist, len(seglist)

	def padding_img(self, img):  # : 1280 720
		img_w = img.shape[1]  # 360 #472
		img_h = img.shape[0]  # 360 #472
		img_c = img.shape[2]  # 360 #472
		pad_img = np.zeros((img_w, img_w, img_c), dtype=np.uint8)
		pad_img.fill(255)

		c_x = img_w / 2
		c_y = img_h / 2

		pad_c_y = img_w / 2
		crop_h = [int(pad_c_y - c_y), int(pad_c_y + c_y)]
		pad_img[crop_h[0]:crop_h[1], :, :] = img

		return pad_img

	def padding_seg(self, img):  # : 1280 720
		img_w = img.shape[1]  # 360 #472
		img_h = img.shape[0]  # 360 #472
		img_c = img.shape[2]  # 360 #472
		pad_seg = np.zeros((img_w, img_w, img_c), dtype=np.uint8)
		pad_seg.fill(0)

		c_x = img_w / 2
		c_y = img_h / 2

		pad_c_y = img_w / 2
		crop_h = [int(pad_c_y - c_y), int(pad_c_y + c_y)]
		pad_seg[crop_h[0]:crop_h[1], :, :] = img

		return pad_seg

	def get_angle(self, cls_idx):
		symm = 0
		pointsList = np.argwhere(self.new_segmented_array == cls_idx)
		pointsList = pointsList - np.mean(pointsList, 0)

		try:
			cov = np.cov(pointsList.transpose())
			evals, evecs = np.linalg.eig(cov)

			# 비슷하면 1, 아니면 0

			if np.abs(np.diff(evals)) <= 5:  # if | eval_0 - eval_1 | < threshold
				symm = 1
			#            if np.abs(np.diff(evals)) <= 10 and cls_idx in [2, 8]:  # if | eval_0 - eval_1 | < threshold
			#                symm = 1
			#            elif np.abs(np.diff(evals)) <= 20 and cls_idx not in [2, 8]:
			#                symm = 1

			if cls_idx in [0, 7]:  # Target : Tape ?  -> 0
				symm = 1

			if symm == 1 and cls_idx in [1, 2, 3, 8]:  # Glue and USB data ignore
				return None, None

			sort_indices = np.argsort(evals)[::-1]
			evec = evecs[sort_indices[1]]
			evec1 = evecs[sort_indices[0]]

			x_v1, y_v1 = evec  # Eigen vector with smallest eigenvalue : short axis
			x_v2, y_v2 = evec1

			theta = np.round(np.arctan(x_v1 / y_v1), 5)  # Radian, Largest Eigen value
			theta1 = np.round(np.arctan(x_v2 / y_v2), 5)

			# TODO : deg/180 ? rad/pi?

			if symm == 1 and cls_idx in [0, 7]:  # sym. object : [Black and White Tape],
				return np.array([0, 1])
			else:
				return np.array([theta, symm])

		except np.linalg.LinAlgError:
			print("def get_angle(self, cls_idx) , np.linalg.LinAlgError :", file=sys.stderr)
			return None, None

	# = 각도 검출
	def get_boxed_angle(self, target_cls):
		cls_points = np.argwhere(self.new_segmented_array == target_cls)  # 새로운 어레이가 타겟 클래스랑 같은 좌표를 찾음
		empty_img = np.zeros((1280, 720), dtype=np.uint8)

		for [x, y] in cls_points:  # 타겟클레스와 같은 점들에 대해
			# empty_img[x, y] = 5         # ?????? 5???
			empty_img[x, y] = 64  # 표시전용 (아무숫자 가능)

		time_str = time.strftime('%Y%m%d-%H-%M-%S', time.localtime(time.time()))
		# >> check
		cv2.namedWindow("empty_img")
		cv2.imshow("empty_img", empty_img)
		# cv2.moveWindow("empty_img", 0, 0)
		cv2.waitKey(2)
		cv2.imwrite("../_/test_img/{}__{}_empty_img".format(time_str, target_cls) + ".png", empty_img)

		# ret, thr = cv2.threshold(empty_img, 1, 29, 0)  # ???? 29 ??????
		ret, thr = cv2.threshold(empty_img, 1, 127, 0)  # 스레스 홀딩했을시 출력할 값

		# >> check
		cv2.namedWindow("thresholding(empty_img)")
		cv2.imshow("thresholding(empty_img)", thr)
		# cv2.moveWindow("thresholding(empty_img)", 2560 - 720 - 256, 340)
		cv2.waitKey(2)
		cv2.imwrite("../_/test_img/{}__{}_thresholding(empty_img)".format(time_str, target_cls) + ".png", thr)

		# con_img, contour, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # ?? 경계선(등고선) 그리기?
		contour, h = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # ?? 경계선(등고선) 그리기?

		# print("finding anlge....")
		for cnt in contour:
			rect = cv2.minAreaRect(cnt)
			angle = abs(rect[2])
			w, h = rect[1]

			box_cont = np.int0(cv2.boxPoints(rect))
			box_img = cv2.drawContours(thr, [box_cont], 0, 255, 1)

			# >> check
			cv2.namedWindow("drawContours(box_img)")
			cv2.imshow("drawContours(box_img)", box_img)  # contour
			# cv2.moveWindow("drawContours(box_img)", 2560 - 720 - 256, 680)
			cv2.waitKey(2)
			cv2.imwrite("../_/test_img/{}__{}_drawContours(box_img)".format(time_str, target_cls) + ".png", box_img)

			if w > h:  # Long axes
				# print("target angle & short axis : %f, %f" % (-90 + angle, h))
				return -90 + angle, h, w
			else:  # Short axes
				# print("target angle & short axis : %f, %f" % (angle, w))
				return angle, w, h

	def getData(self, target_cls):
		try:
			pointsList = np.argwhere(self.new_segmented_array == target_cls)
			mean_pt = np.copy(np.mean(pointsList, 0))

			return pointsList, mean_pt

		except np.linalg.LinAlgError:
			print("np.linalg.LinAlgError", file=sys.stderr)
			return None, None

	def convert_grey_label_to_color_label(self, grey_label_array):
		height = grey_label_array.shape[0]
		width = grey_label_array.shape[1]
		channel = 3

		shape = (height, width, channel)
		color_label = np.zeros(shape=shape, dtype=np.uint8)

		for i in range(0, height):
			for j in range(0, width):
				value = grey_label_array[i, j]
				color = RL_Obj_List[value][1]

				color_label[i, j, 0] = color[0]
				color_label[i, j, 1] = color[1]
				color_label[i, j, 2] = color[2]

		return np.copy(color_label)

	def make_binary_label_array(self, target_object_pixels_list, binary_array):
		num_points = len(target_object_pixels_list)

		for i in range(0, num_points):
			pixel = target_object_pixels_list[i]
			y = pixel[0]
			x = pixel[1]
			binary_array[y, x] = 255

		return binary_array

	def make_new_label_array(self, dest_array, target_object_pixels_list, object_index):
		num_points = len(target_object_pixels_list)

		for i in range(0, num_points):
			pixel = target_object_pixels_list[i]
			y = pixel[0]
			x = pixel[1]
			dest_array[y, x] = object_index

		return dest_array

	def getPoints(self, label_image_array, object_index):
		return np.argwhere(label_image_array == object_index)

	def new_segmented_image_by_connected_component(self, org_label_array):
		shape = (1280, 720)
		binary_image_array = np.zeros(shape=shape, dtype=np.uint8)
		new_label_image_array = np.zeros(shape=shape, dtype=np.uint8)
		new_label_image_array.fill(0)

		for j in range(0, 31):
			pointList = self.getPoints(org_label_array, j)
			binary_image_array.fill(0)
			binary_image_array = self.make_binary_label_array(pointList, binary_image_array)

			connectivity = 8
			num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image_array, connectivity,
			                                                                cv2.CV_32S)
			# numlabel : The first cell is the number of labels
			# The second cell is the label matrix

			if num_labels > 1:
				second_max_index = 1
				second_max = 0
				for m in range(0, num_labels):
					pixels_num = stats[m, cv2.CC_STAT_AREA]
					if second_max < pixels_num < 10000:
						second_max = pixels_num
						second_max_index = m

				pointList2 = self.getPoints(labels, second_max_index)

				new_label_image_array = self.make_new_label_array(new_label_image_array, pointList2, j)
			else:
				if len(pointList) > 0:
					new_label_image_array = self.make_new_label_array(new_label_image_array, pointList, j)

		# Gray Seg.
		new_label_image_array = self.smoothing(new_label_image_array)

		# Color Seg.
		new_label_color_image_array = self.convert_grey_label_to_color_label(new_label_image_array)

		return new_label_image_array, new_label_color_image_array

	@staticmethod
	def smoothing(seg):
		obj_unique, pix_count = np.unique(seg, return_counts=True)

		for idx, cnt in enumerate(pix_count):
			if cnt < 20:
				xy = np.argwhere(seg == obj_unique[idx])

				for x, y in xy:
					seg[x, y] = 0

		return seg
