#!/usr/bin/env python3

import cv2
import threading
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from yolo_msgs.msg import Inference

bridge = CvBridge()

class Camera_subscriber(Node):

    def __init__(self):
        super().__init__('camera_subscriber')
        self.subscription = self.create_subscription(
            Image,
            'camera/color/image_raw', #'camera/image_raw', for gazebo simulation
            self.listener_callback,
            10)
        self.subscription

    def camera_callback(self, data):
        global img
        img = bridge.imgmsg_to_cv2(data, "bgr8")

class Yolo_Subscriber(Node):

    def __init__(self):
        super().__init__

        self.subscription = self.create_subscription(
            Inference,
            'yolo/inference',
            self.yolo_callback,
            10)
        self.subscription

        self.cnt = 0

        self.img_pub = self.create_publisher(Image, 'yolo/image', 1)

    def yolo_callback(self, data):
        global img
        for r in data.yolov8_inference:

            class_name = r.class_name
            top = r.top
            left = r.left
            bottom = r.bottom
            right = r.right
            yolo_subscriber.get_logger().info(f"{self.cnt} {class_name} : {top},{left},{bottom},{right}")
            cv2.rectangle(img,(top, left), (bottom, right), (0, 255, 0), 2)
            self.cnt += 1

        self.cnt = 0
        img_msg = bridge.cv2_to_imgmsg(img, "bgr8")
        self.img_pub.publish(img_msg)

    if __name__ == '__main__':
        rclpy.init(args=None)
        yolo_subscriber = Yolo_Subscriber()
        camera_subscriber = Camera_subscriber()

        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(yolo_subscriber)
        executor.add_node(camera_subscriber)

        executor_thread = threading.Thread(target=executor.spin, daemon=True)

        rate = yolo_subscriber.create_rate(2)
        try :
            while rclpy.ok():
                rate.sleep()

        except KeyboardInterrupt:
            pass

        rclpy.shutdown()
        executor_thread.join()

