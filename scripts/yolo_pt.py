#!/usr/bin/env python3

from ultralytics import YOLO
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from yolo_msgs.msg import Inference
from yolo_msgs.msg import InferenceResult

bridge = CvBridge()

class Camera_subscriber(Node):

    def __init__(self):
        super().__init__('camera_subscriber')

        self.model = YOLO('yolov8n.pt')

        self.yolo_inference = Inference()

        self.subscription = self.create_subscription(
            Image,
            'camera/color/image_raw', #'camera/image_raw', for gazebo simulation
            self.camera_callback,
            10)
        self.subscription

        self.yolo_pub = self.create_publisher(Inference, 'yolo/inference', 1)
        self.img_pub = self.create_publisher(Image, 'yolo/image', 1)

    
    def camera_callback(self, data):
        img = bridge.imgmsg_to_cv2(data, "bgr8")    
        results = self.model(img)

        self.yolo_inference.header.frame_id = 'inference'
        self.yolo_inference.header.stamp = camera_subscriber.get_clock().now().to_msg()

        for r in results :
            boxes = r.boxes
            for box in boxes :
                self.inference_result = InferenceResult()
                b = box.xyxy[0].to('cpu').detach().numpy().copy() # b = box.xyxy[0].to('cuda:0').detach().numpy().copy()
                 # get box coordinates in (top, left ,bottom,right)
                c = box.cls
                self.inference_result.class_name = self.model.names[int(c)]
                self.inference_result.top = int(b[0])
                self.inference_result.left = int(b[1])
                self.inference_result.bottom = int(b[2])
                self.inference_result.right = int(b[3])

        annotated_frame = results[0].plot()
        img_msg = bridge.cv2_to_imgmsg(annotated_frame)

        self.img_pub.publish(img_msg)
        self.yolo_pub.publish(self.yolo_inference)
        self.yolo_inference.yolo_inference.clear()

        
if __name__ == "__main__":
    rclpy.init()
    camera_subscriber = Camera_subscriber()
    rclpy.spin(camera_subscriber)
    rclpy.shutdown()