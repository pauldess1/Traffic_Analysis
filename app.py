import numpy as np
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

class VideoProcessor():
    def __init__(self, video_in_path, video_out_path, model):
        self.video_in_path = video_in_path
        self.video_out_path = video_out_path
        self.model = model
    
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
        for frame in frames:
            results = self.model(frame)
            boxes = results[0].boxes.xywh.numpy()
            for box in boxes:
                x, y, w, h = box
                cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 0), 2)
            processed_frames.append(frame)
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
    video_in_path = "data/video_cut3.mp4"
    video_out_path = "data/processed_video_cut3.mp4"
    model = YOLO("traffic_analysis.pt")
    processor = VideoProcessor(video_in_path, video_out_path, model)

    listOfFrames = processor.video_to_frames()
    processed_frames = processor.process_frames(listOfFrames)
    processor.save_processed_video(processed_frames)

main()