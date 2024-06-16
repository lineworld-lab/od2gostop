#! /usr/bin/env python3
import cv2 as cv
import numpy as np
import pandas as pd
import torch
import sys
import sklearn
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ultralytics import YOLO

from shapely.geometry import Point, Polygon
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
from yolo_msgs.msg import Inference
from yolo_msgs.msg import InferenceResult

bridge = CvBridge()


class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('camera_subscriber')

        self.model = YOLO('yolov8n.pt')
        self.yolo_inference = Inference()

        self.subscription = self.create_subscription(
            Image,
            'camera/color/image_raw',  # 'camera/image_raw', for gazebo simulation
            self.camera_callback,
            10)
        self.subscription

        self.yolo_pub = self.create_publisher(Inference, 'yolo/inference', 1)
        self.img_pub = self.create_publisher(Image, 'yolo/image', 1)



    def camera_callback(self, data):

        def point_in_polygon (point,polygon):
            point = Point(point)
            polygon = Polygon(polygon)
            if polygon.contains(point):
                return "invade"
            else :
                return "not_invade"
    

        img = bridge.imgmsg_to_cv2(data, "bgr8")    
        results = self.model(img)

        self.yolo_inference.header.frame_id = 'inference'
        self.yolo_inference.header.stamp = self.get_clock().now().to_msg()

        points = [(0, 0), (300, 0), (300, 300), (0, 300)]
        points_mask = np.array([[0,0], [300, 0], [300, 300], [0, 300]])
        polygon = Polygon(points)
        points_arr = np.array(points)

    ## gaussian blur
      #  mask = np.zeros_like(img)

      #  cv.fillPoly(mask, [points_mask], 1)

      #  blur = cv.GaussianBlur(img, (99, 99), 0)
      #  img = img * mask[:,:,None] + blur * (1 - mask[:,:,None])
      #  img = img * mask + blur * (1 - mask)

       # cv.polylines(img, [points_arr.astype(np.int32)], True, (0, 255, 255), 2)

        for r in results :
            boxes = r.boxes
            for box in boxes :
        # Get the bounding box coordinates
                b = box.xyxy[0].to('cpu').detach().numpy().copy().astype(int)
                ct = box.xywh[0].to('cpu').detach().numpy().copy().astype(int)
        # Calculate the center coordinates of the bounding box
        # Check if the center is within the polygon
                if point_in_polygon(ct[:2], points) == "invade":
            # This object is within the polygon
                    self.inference_result = InferenceResult()
                    c = box.cls
                    self.inference_result.class_name = self.model.names[int(c)]
                    self.inference_result.top = int(b[0])
                    self.inference_result.left = int(b[1])
                    self.inference_result.bottom = int(b[2])
                    self.inference_result.right = int(b[3])

                    self.yolo_inference.yolo_inference.append(self.inference_result)

        annotated_frame = results[0].plot()

        cv.polylines(annotated_frame, [points_arr.astype(np.int32)], True, (0, 255, 255), 2)
        img_msg = bridge.cv2_to_imgmsg(annotated_frame)

        self.img_pub.publish(img_msg)
        self.yolo_pub.publish(self.yolo_inference)
        self.yolo_inference.yolo_inference.clear()

                    
if __name__ == "__main__":
    rclpy.init()
    camera_subscriber = CameraSubscriber()
    rclpy.spin(camera_subscriber)
    rclpy.shutdown()