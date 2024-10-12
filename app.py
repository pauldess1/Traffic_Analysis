import numpy as np
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import math

class Tracker():
        def __init__(self):
            tracking_objects = {}
            center_points_cur_frame = []
            center_points_prv_frame = []
            track_id = 0

            self.tracking_objects = tracking_objects
            self.center_points_cur_frame = center_points_cur_frame
            self.center_points_prv_frame = center_points_prv_frame
            self.track_id = track_id

        def tracking(self):
            for track_id, prv in self.tracking_objects.items():
                found_match = False

                for cur in self.center_points_cur_frame:
                    distance = math.hypot(prv[0] - cur[0], prv[1] - cur[1])

                    if distance < 20:
                        self.tracking_objects[track_id] = cur
                        found_match = True
                        break

                    if not found_match:
                        pass

            for cur in self.center_points_cur_frame:
                found_existing = False

                for prv in self.tracking_objects.values():
                    distance = math.hypot(prv[0] - cur[0], prv[1] - cur[1])

                    if distance < 20:
                        found_existing = True
                        break

                if not found_existing:
                    self.tracking_objects[self.track_id] = cur
                    self.track_id += 1

        def update(self, new_center_points_frame):
            self.center_points_prv_frame = self.center_points_cur_frame.copy()
            self.center_points_cur_frame = new_center_points_frame.copy()
            self.tracking()



class VideoProcessor():
    def __init__(self, video_in_path, video_out_path, model):
        self.video_in_path = video_in_path
        self.video_out_path = video_out_path
        self.model = model
        self.tracker = Tracker()
    
    def video_to_frames(self):
        frames = []
        cap = cv2.VideoCapture(self.video_in_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    def process_frames(self, frames):
        processed_frames = []
        nb = 0
        for frame in frames:
            print(nb)
            center_points = []
            results = self.model(frame)
            boxes = results[0].boxes.xywh.numpy()
            for box in boxes:
                x, y, w, h = box
                center_points.append((int(x),int(y)))
                cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 0), 2)
            
            self.tracker.update(center_points)
            processed_frames.append(frame)

            for object_id, cur in self.tracker.tracking_objects.items():
                cv2.circle(frame, cur, 5, (0, 0, 255), -1)
                cv2.putText(frame, str(object_id), (cur[0], cur[1]-5), 0, 1, (0, 0, 255), 2)
            nb+=1
        return processed_frames

    def save_processed_video(self, processed_frames):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30
        height, width, _ = processed_frames[0].shape
        out = cv2.VideoWriter(self.video_out_path, fourcc, fps, (width, height))
        
        for frame in processed_frames:
            out.write(frame)
        out.release()

def main():
    video_in_path = "data/video.mov"
    video_out_path = "data/processed_video.mp4"
    model = YOLO("traffic_analysis.pt")
    processor = VideoProcessor(video_in_path, video_out_path, model)

    listOfFrames = processor.video_to_frames()
    processed_frames = processor.process_frames(listOfFrames)
    processor.save_processed_video(processed_frames)

main()

