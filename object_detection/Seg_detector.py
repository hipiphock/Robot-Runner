import numpy as np
import sys
import tensorflow as tf
import cv2
from Robot_env.config import RL_Obj_List
import time

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

sys.path.append("..")
PATH_TO_FROZEN_GRAPH = './object_detection/frozen_graph/frozen_inference_graph.pb'
PATH_TO_LABELS = './object_detection/frozen_graph/labelmap.pbtxt'
# PATH_TO_FROZEN_GRAPH = './object_detection/196259/frozen_inference_graph.pb'
# PATH_TO_LABELS = './object_detection/196259/labelmap.pbtxt'

NUM_CLASSES = 41
IMAGE_CHANNEL = 3


class Segment:

    def __init__(self):

        self.detection_graph = tf.Graph()

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()

            with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}

            self.tensor_dict = {}
            for key in ['num_detections',
                        'detection_boxes',
                        'detection_scores',
                        'detection_classes',
                        'detection_masks']:
                tensor_name = key + ':0'

                if tensor_name in all_tensor_names:
                    self.tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

                if 'detection_masks' in self.tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(self.tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(self.tensor_dict['detection_masks'], [0])
                    real_num_detection = tf.cast(self.tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, 720, 1280)
                    detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    self.tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)

            self.image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            self.sess = tf.Session()

        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

    def run_inference_for_single_image(self, image):
        output_dict = self.sess.run(self.tensor_dict, feed_dict={self.image_tensor: np.expand_dims(image, 0)})
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict

    def visualize(self, image, boxes, classes, scores, masks, threshold):
        image = vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            boxes,
            classes,
            scores,
            self.category_index,
            instance_masks=masks,
            use_normalized_coordinates=True,
            line_thickness=2,
            min_score_thresh=threshold
        )
        return image

    def model_run(self, image):
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        output_dict = self.run_inference_for_single_image(image_np)
        boxes = output_dict['detection_boxes']
        classes = output_dict['detection_classes']
        scores = output_dict['detection_scores']
        masks = output_dict.get('detection_masks')
        return boxes, classes, scores, masks

    def get_seg(self, classes, scores, masks, threshold):
        idx = np.where(scores >= threshold)
        classes = classes[idx]
        masks = masks[idx]
        width = len(masks[0][0])
        height = len(masks[0])
        seg = np.zeros((height, width), dtype=np.uint8)
        for i in range(len(classes)):
            id_ = classes[i]
            idx = np.where(masks[i] != 0)
            seg[idx] = id_
        return seg

    def total_run(self, image, threshold=0.60):
        t1 = time.time()
        image_copy = image.copy()
        # image_copybgr = cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR)
        image_copybgr = image.copy()[..., ::-1]
        output_dict = self.run_inference_for_single_image(image_copybgr)
        boxes = output_dict['detection_boxes']
        classes = output_dict['detection_classes']
        scores = output_dict['detection_scores']
        masks = output_dict.get('detection_masks')

        image_vis = vis_util.visualize_boxes_and_labels_on_image_array(image, boxes, classes, scores,
                                                                       self.category_index,
                                                                       instance_masks=masks,
                                                                       use_normalized_coordinates=True,
                                                                       # max_boxes_to_draw=30,
                                                                       max_boxes_to_draw=30,
                                                                       line_thickness=2,
                                                                       min_score_thresh=threshold)

        idx = np.where(scores >= threshold)
        classes = classes[idx]
        masks = masks[idx]
        # width = len(masks[0][0])
        # height = len(masks[0])
        [obj_masknum, height, width] = masks.shape

        seg = np.zeros((height, width), dtype=np.uint8)
        # for i in range(len(classes)):
        #     id = classes[i]
        #     idx = np.where(masks[i] != 0)
        #     seg[idx] = id
        for x, i in enumerate(classes):
            idx = np.where(masks[x] != 0)
            seg[idx] = i

        color_seg = self.convert_color_seg(seg)

        t2 = time.time()
        print("time spend : {:07.4f} ... total_run".format(t2-t1))

        return seg, color_seg, None

    def convert_color_seg(self, grey_label_array):
        t1 = time.time()

        height = grey_label_array.shape[0]
        width = grey_label_array.shape[1]
        channel = IMAGE_CHANNEL

        shape = (height, width, channel)
        color_label = np.zeros(shape=shape, dtype=np.uint8)

        # for i in range(0, height):
        #     for j in range(0, width):
        #         value = grey_label_array[i, j]
        #         color = RL_Obj_List[value][1]
        #
        #         color_label[i, j] = color
        for i in np.unique(grey_label_array):
            list0 = np.where(grey_label_array == i)
            color = RL_Obj_List[i][1]
            color_label[list0[0], list0[1]] = color

        t2 = time.time()

        print("time spend : {:07.4f} ... convert_color_seg".format(t2-t1))
        return np.copy(color_label)

