import numpy as np
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import math
import argparse
import csv
import pandas as pd
import os
import seaborn as sns

class Tracker():
    def __init__(self):
        self.tracking_objects = {}
        self.center_points_cur_frame = []
        self.center_points_prv_frame = []
        self.track_id = 0
        self.trajectories = {}

    def tracking(self):
        for track_id, prv in list(self.tracking_objects.items()):
            found_match = False

            for cur in self.center_points_cur_frame:
                distance = math.hypot(prv[0] - cur[0], prv[1] - cur[1])

                if distance < 20:
                    self.tracking_objects[track_id] = cur
                    found_match = True
                    
                    if track_id not in self.trajectories:
                        self.trajectories[track_id] = []
                    self.trajectories[track_id].append(cur)
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
                self.trajectories[self.track_id] = [cur]
                self.track_id += 1

    def update(self, new_center_points_frame):
        self.center_points_prv_frame = self.center_points_cur_frame.copy()
        self.center_points_cur_frame = new_center_points_frame.copy()
        self.tracking()

    def load_trajectories(self, csv_file):
        data = pd.read_csv(csv_file)
        for track_id in data['track_id'].unique():
            self.tracking_objects[track_id] = data[data['track_id'] == track_id].iloc[0][['x', 'y']].values.tolist()
            self.trajectories[track_id] = data[data['track_id'] == track_id][['x', 'y']].values.tolist()
            self.track_id = max(self.track_id, track_id + 1)
    
    def save_trajectories(self, file_path):
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["track_id", "x", "y"])

            for track_id, trajectory in self.trajectories.items():
                for x, y in trajectory:
                    writer.writerow([track_id, x, y])


class VideoProcessor():
    def __init__(self, video_in_path, video_out_path, model):
        self.video_in_path = video_in_path
        self.video_out_path = video_out_path
        self.model = model
        self.tracker = Tracker()
        self.shape = None
    
    def video_to_frames(self):
        frames = []
        cap = cv2.VideoCapture(self.video_in_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        self.shape = frames[0].shape
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

class Visualizer:
    def __init__(self, tracker, frame_shape):
        self.tracker = tracker
        self.frame_shape = frame_shape

    def load_trajectories(self, csv_file):
        try:
            data = pd.read_csv(csv_file)
            trajectories = {}
            
            for track_id in data['track_id'].unique():
                trajectories[track_id] = data[data['track_id'] == track_id][['x', 'y']].values.tolist()
                
            return trajectories
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return {}

    def plot_trajectories(self):
        plt.figure(figsize=(10, 6))
        for track_id, trajectory in self.tracker.trajectories.items():
            x_vals = [pos[0] for pos in trajectory]
            y_vals = [pos[1] for pos in trajectory]
            plt.plot(x_vals, y_vals, marker='o', linestyle='-')

        plt.xlabel("Position X")
        plt.ylabel("Position Y")
        plt.title("Vehicle trajectories")
        plt.grid(True)

        plt.xlim(0, 1920)
        plt.ylim(1080, 0)

        plt.show()

    def plot_heatmap(self):
        all_positions = []
        for trajectory in self.tracker.trajectories.values():
            all_positions.extend(trajectory)
        positions_array = np.array(all_positions)

        plt.figure(figsize=(10, 6))
        sns.kdeplot(x=positions_array[:, 0], y=positions_array[:, 1], cmap='viridis', fill=True, thresh=0, levels=100)
        
        plt.xlabel("Position X")
        plt.ylabel("Position Y")
        plt.title("Vehicle position heatmap")
        plt.xlim(0, 1920)
        plt.ylim(1080, 0)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def run(self):
        self.plot_trajectories()
        self.plot_heatmap()

def main():
    parser = argparse.ArgumentParser(description="Processes a video to create trajectories or visualizes trajectories from a CSV file.")
    parser.add_argument('--video_in', type=str, help="Path to the input video", default="data/in/video.mov")
    parser.add_argument('--video_out', type=str, help="Path to the output video", default="data/out/video_complete.mp4")
    parser.add_argument('--model_path', type=str, default="traffic_analysis.pt", help="Path to the YOLO model to use")
    parser.add_argument('--csv_file', type=str, help="Path to the CSV file containing the trajectories", default="data/out/trajectories.csv")

    args = parser.parse_args()

    csv_file_path = args.csv_file

    if not os.path.exists(csv_file_path):
        video_in_path = args.video_in
        video_out_path = args.video_out
        model_path = args.model_path

        if not os.path.exists(video_in_path):
            print(f"Error: The input video file '{video_in_path}' does not exist.")
            return
        
        if not os.path.exists(model_path):
            print(f"Error: The model '{model_path}' does not exist.")
            return

        model = YOLO(model_path)
        processor = VideoProcessor(video_in_path, video_out_path, model)

        listOfFrames = processor.video_to_frames()
        if processor.shape is None:
            print("Error: Unable to retrieve the video's shape.")
            return

        processed_frames = processor.process_frames(listOfFrames)
        processor.save_processed_video(processed_frames)

        processor.tracker.save_trajectories(csv_file_path)
        print(f"Trajectories saved in '{csv_file_path}'.")

    else:
        print(f"The CSV file '{csv_file_path}' already exists. Loading trajectories...")
    
    visu = Visualizer(tracker=Tracker(), frame_shape=(720, 1280, 3))
    visu.tracker.load_trajectories(csv_file_path)
    visu.run()

if __name__ == "__main__":
    main()
