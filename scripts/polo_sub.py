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
from geometry_msgs.msg import Point32, Polygon
from cv_bridge import CvBridge

from ultralytics import YOLO
from collections import Counter, deque
from shapely.geometry import Polygon, Point
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
from yolo_msgs.msg import Inference, InferenceResult



class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('camera_subscriber')

        self.bridge = CvBridge()

        
        self.model = YOLO('yolov8n.pt')
        self.yolo_inference = Inference()

        self.subscription = self.create_subscription(
            Image,
            'camera/color/image_raw',  # 'camera/image_raw', for gazebo simulation
            self.camera_callback,
            10)
        self.subscription
        print("successfully initialized")
        self.yolo_pub = self.create_publisher(Inference, 'yolo/inference', 1)
        self.img_pub = self.create_publisher(Image, 'yolo/image', 1)
    
    def blur_img(self, img, kernel_size):
        kernel_size = max(1, int(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1

        blured_img = cv.blur(img, (kernel_size, kernel_size))
        return blured_img

    def camera_callback(self, data):
        img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        annotated_frame = None  # 변수 초기화
        points = [(0, 0), (400, 0), (400, 640), (0, 640)]
        ## polygon msg testing code
        
        # points = [Point32(x=0, y=0, z=0), Point32(x=400, y=0, z=0), Point32(x=400, y=640, z=0), Point32(x=0, y=640, z=0)]
        # polygon = Polygon(points)

        # self.polygon_pub = self.create_publisher(Polygon, 'yolo/polygon', 1)
        # self.polygon_pub.publish(polygon)

        points_arr = np.array(points)
        cv.polylines(img, [points_arr.astype(np.int32)], True, (0, 255, 255), 2)

        def point_in_polygon(point, polygon):
            point = Point(point)
            polygon = Polygon(polygon)

            if polygon.contains(point):
                return "invade"
            else:
                return "safe"

        frame_quality = 1
        psnr_values = []
        Cnt = []
        ground_truth_label = []
        df = pd.DataFrame(columns=['CNT', '0', 'invade', 'safe'])
        ground_truth_label = 0
        predict_labels = []

        for fq in range(frame_quality):
            df_con = pd.DataFrame(columns=['invade', 'safe'])
            count = 0

            log_path = "/usr/tmp/realsense/"
            
            width, height = img.shape[:2]

            fps = 30
            
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            output_path = f'{log_path}output_{fq}.mp4'
            out = cv.VideoWriter(output_path, fourcc, fps, (width, height))
            cv.namedWindow(f'{fq}')
            invade_check = deque(maxlen=3)

            while True:
                count += 1
                invade_frame = 0
                frame = img.copy()

                frame_c = img.copy()
                frame_degrade = self.blur_img(img, fq)

                psnr = cv.PSNR(frame_c, frame_degrade)
                psnr_values.append(psnr)

                mask = np.zeros(frame_degrade.shape)
                if len(points) > 1:
                    points_arr = np.array(points)
                    cv.polylines(frame_degrade, [points_arr.astype(np.int32)], True, (0, 255, 255), 2)

                results = self.model.predict(frame_degrade, verbose=False, conf=0.5)[0]
                boxes = results.boxes.cpu().numpy()

                if len(points) > 3:
                    class_check = []
                    for r in results:
                        boxes = r.boxes
                        for box in boxes:
                            self.inference_result = InferenceResult()
                            b = box.xyxy[0].to('cpu').detach().numpy().copy()  # b = box.xyxy[0].to('cuda:0').detach().numpy().copy()
                            c = box.cls
                            class_name = point_in_polygon(c[:2],points)
                            self.inference_result.class_name = self.model.names[int(c)]
                            self.inference_result.top = int(b[0])
                            self.inference_result.left = int(b[1])
                            self.inference_result.bottom = int(b[2])
                            self.inference_result.right = int(b[3])

                            cv.rectangle(frame_degrade, r[:2], r[:2], (255, 255, 255), 3)
                            confi = np.round(box.conf[0], 4)

                            cv.putText(frame_degrade, f"{class_name}/{confi:.3f}", cv.FONT_HERSHEY_SIMPLEX)
                            class_check.append(class_name)
                            predict_labels.append(class_name)

                        annotated_frame = results[0].plot()

                else:
                    class_check = []
                    for r in results:
                        boxes = r.boxes
                        for box in boxes:
                            self.inference_result = InferenceResult()
                            b = box.xyxy[0].to('cpu').detach().numpy().copy()
                            c = box.cls
                            class_name = "safe"
                            self.inference_result.class_name = self.model.names[int(c)]
                            self.inference_result.top = int(b[0])
                            self.inference_result.left = int(b[1])
                            self.inference_result.bottom = int(b[2])
                            self.inference_result.right = int(b[3])

                            cv.rectangle(frame_degrade, r[:2], r[:2], (255, 255, 255), 3)
                            confi = np.round(box.conf[0], 4)
                            cv.putText(frame_degrade, f"{class_name}/{confi:.3f}", cv.FONT_HERSHEY_SIMPLEX)
                            class_check.append(class_name)
                            predict_labels.append(class_name)

                        annotated_frame = results[0].plot()

                    if len(results.boxes) == 0:
                        predict_labels.append("0")

                    class_counts = Counter(class_check)
                    if 'invade' in class_check:
                        invade_frame = "invade"
                    invade_check.append(invade_frame)

                    if fq == 1:
                        ground_truth_label = predict_labels

                    df_counter = pd.DataFrame.from_dict(Counter(class_check), orient='index', columns=['Count']).T
                    df_counter = df_counter.reset_index(drop=True)

                    df_con = pd.concat([df_con, df_counter], axis=0, ignore_index=True)

                    out.write(frame_degrade)

                cd = Counter(predict_labels)
                df_count = pd.DataFrame({'count': [count],
                                         'PSNR': [sum(psnr_values) / len(psnr_values)]})
                df3 = pd.concat([df_count], axis=1)
                df = pd.concat([df, df3], axis=0, ignore_index=True)

                out.release()
                cv.destroyAllWindows()

        if annotated_frame is not None:  # 변수가 할당되었는지 확인
            img_msg = bridge.cv2_to_imgmsg(annotated_frame)
            self.img_pub.publish(img_msg)
            self.yolo_pub.publish(self.yolo_inference)
            self.yolo_inference.yolo_inference.clear()

            
if __name__ =="__main__" :
    rclpy.init()
    camera_subscriber = CameraSubscriber()
    rclpy.spin(camera_subscriber)
    rclpy.shutdown()

