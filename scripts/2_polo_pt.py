#!/usr/bin/env python3

import cv2 as cv
import numpy as np
import pandas as pd

import os
import timeit
import sklearn
import sys

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

from ultralytics import YOLO
from collections import Counter, deque
from shapely.geometry import Point
from shapely.geometry import Polygon
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
from yolo_msgs.msg import Inference
from yolo_msgs.msg import InferenceResult

bridge = CvBridge()


class CameraSubscriber(Node):

    def __init__(self):
        super().__init__('camera_subscriber')
        self.get_logger().info('Polo Node is started')
        self.model = YOLO('yolov8n.pt')
        self.yolo_inference = Inference()

        # self.subscription = self.create_subscription(
        #     Image,
        #     'camera/color/image_raw',
        #     self.camera_callback,
        #     10)
        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.camera_callback,
            10)
        self.subscription

        self.get_logger().debug('Subscribed to camera topic')

        ## published messages
        self.yolo_pub = self.create_publisher(Inference, 'yolo/inference', 1)
        self.img_pub = self.create_publisher(Image, 'yolo/image', 1)
        self.twist_pub = self.create_publisher(Twist, 'cmd_vel_invade', 1)

        self.points = [(0, 0), (300, 0), (300, 300), (0, 300)]
        self.points_arr = np.array(self.points)
        self.polygon = Polygon(self.points)

    def point_in_polygon(self, point):
        point = Point(point)
        return self.polygon.contains(point)

    def camera_callback(self, data):

        self.yolo_inference.header.frame_id = 'inference'
        self.yolo_inference.header.stamp = self.get_clock().now().to_msg()

        img = bridge.imgmsg_to_cv2(data, "bgr8")

        results = self.model(img)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                self.inference_result = InferenceResult()

                r = box.xyxy[0].cpu().numpy().astype(int)
                ct = box.xywh[0].cpu().numpy().astype(int)
                inside_polygon = self.point_in_polygon(ct[:2])
                if inside_polygon:  # Only draw bounding box and class name if the object is inside the polygon
                    class_name = 'INVADE'  # replace with your actual class name
                    cv.rectangle(img, (r[0], r[1]), (r[2], r[3]), (255, 255, 255), 2)
                    confi = np.round(box.conf[0].cpu().numpy(), 4)
                    cv.putText(img, f"{class_name}/ {confi:.3f}", (r[0], r[1] - 10), cv.FONT_HERSHEY_SIMPLEX,
                            .5, (0, 255, 0), 2, cv.LINE_AA)

                    self.inference_result.class_name = class_name
                    self.inference_result.top = int(r[0])
                    self.inference_result.left = int(r[1])
                    self.inference_result.bottom = int(r[2])
                    self.inference_result.right = int(r[3])

                    self.get_logger().debug(f"{len(class_name)}object invade the polygon at {ct[:2]} with confidence {confi}")
                    self.yolo_inference.yolo_inference.append(self.inference_result)
                    # emergency stop if object is detected
                    twist = Twist()
                    twist.linear.x = 0.0
                    twist.linear.y = 0.0
                    twist.linear.z = 0.0
                    twist.angular.x = 0.0
                    twist.angular.y = 0.0
                    twist.angular.z = 0.0
                    self.twist_pub.publish(twist)
                    self.get_logger().info("Emergency stop!!!")
                    self.get_logger().debug(f"{twist}")
        cv.polylines(img, [self.points_arr.astype(np.int32)], True, (0, 255, 255), 2)
        img_msg = bridge.cv2_to_imgmsg(img)

        self.img_pub.publish(img_msg)
        self.yolo_pub.publish(self.yolo_inference)
        self.yolo_inference.yolo_inference.clear()


if __name__ == "__main__":
    rclpy.init()
    camera_subscriber = CameraSubscriber()
    rclpy.spin(camera_subscriber)
    rclpy.shutdown()