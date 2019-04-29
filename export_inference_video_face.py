#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

import sys
import time
import numpy as np
import tensorflow as tf
import cv2

sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils_color as vis_util


import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.python.framework import tensor_util


# Path to frozen detection graph. This is the actual model that is used
# for the object detection.
PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './protos/face_label_map.pbtxt'

NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

cap = cv2.VideoCapture("./media/test.mp4")
out = None


frame_num = 1490
while frame_num:
    frame_num -= 1
    ret, image = cap.read()
    if ret == 0:
        break

    if out is None:
        [h, w] = image.shape[:2]
        out = cv2.VideoWriter("./media/test_out.avi", 0, 25.0, (w, h))

    image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    # Expand dimensions since the model expects images to have shape:
    # [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)

    # image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # # Each box represents a part of the image where a particular object was detected.
    # boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # # Each score represent how level of confidence for each of the objects.
    # # Score is shown on the result image, together with the class label.
    # scores = detection_graph.get_tensor_by_name('detection_scores:0')
    # classes = detection_graph.get_tensor_by_name('detection_classes:0')
    # num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # export_path = "/tmp/face_detector/0"
    # print('Exporting trained model to', export_path)

    # builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    # # Define input tensor
    # image_tensor_serving = tf.saved_model.utils.build_tensor_info(image_tensor)

    # # Define output tensor
    # boxes_serving = tf.saved_model.utils.build_tensor_info(boxes)
    # scores_serving = tf.saved_model.utils.build_tensor_info(scores)
    # classes_serving = tf.saved_model.utils.build_tensor_info(classes)
    # num_detections_serving = tf.saved_model.utils.build_tensor_info(num_detections)

    # prediction_signature = (
    #    tf.saved_model.signature_def_utils.build_signature_def(
    #        inputs={'image_tensor': image_tensor_serving},
    #        outputs={'boxes': boxes_serving, 'scores': scores_serving, 'classes': classes_serving, 'num_detections': num_detections_serving},
    #        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    # builder.add_meta_graph_and_variables(
    #     sess, [tf.saved_model.tag_constants.SERVING],
    #     signature_def_map={
    #         'predict_output':
    #             prediction_signature,
    #     },
    #     main_op=tf.tables_initializer(),
    #     strip_default_attrs=True)

    # builder.save()

    # Actual detection.
    start_time = time.time()

    channel = grpc.insecure_channel('0.0.0.0:8500')
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'face_detector'
    request.model_spec.signature_name = 'predict_output'
    request.inputs['image_tensor'].CopyFrom(
        tf.contrib.util.make_tensor_proto(image_np_expanded, shape=list(image_np_expanded.shape)))

    result = stub.Predict(request, 10.0)  # 5 seconds

    boxes = tensor_util.MakeNdarray(
        result.outputs['boxes'])

    scores = tensor_util.MakeNdarray(
        result.outputs['scores'])

    classes = tensor_util.MakeNdarray(
        result.outputs['classes'])

    num_detections = tensor_util.MakeNdarray(
        result.outputs['num_detections'])

    elapsed_time = time.time() - start_time
    print('inference time cost: {}'.format(elapsed_time))
    #print(boxes.shape, boxes)
    # print(scores.shape,scores)
    # print(classes.shape,classes)
    # print(num_detections)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        #          image_np,
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=4)
    out.write(image)

cap.release()
out.release()
