import numpy as np
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import math

class Tracker():
    def __init__(self):
        self.tracking_objects = {}
        self.center_points_cur_frame = []
        self.center_points_prv_frame = []
        self.track_id = 0
        self.trajectories = {}  # Enregistrement des trajectoires

    def tracking(self):
        # Mettre à jour chaque objet suivi
        for track_id, prv in list(self.tracking_objects.items()):
            found_match = False

            for cur in self.center_points_cur_frame:
                distance = math.hypot(prv[0] - cur[0], prv[1] - cur[1])

                if distance < 20:
                    # Mise à jour de la position de l'objet suivi
                    self.tracking_objects[track_id] = cur
                    found_match = True
                    
                    # Enregistrer la position dans les trajectoires
                    if track_id not in self.trajectories:
                        self.trajectories[track_id] = []
                    self.trajectories[track_id].append(cur)
                    break

            if not found_match:
                # Si aucun match n'est trouvé, ne pas supprimer immédiatement
                pass

        # Ajouter de nouveaux objets si aucun match n'est trouvé
        for cur in self.center_points_cur_frame:
            found_existing = False

            for prv in self.tracking_objects.values():
                distance = math.hypot(prv[0] - cur[0], prv[1] - cur[1])

                if distance < 20:
                    found_existing = True
                    break

            if not found_existing:
                # Ajouter un nouvel objet et initialiser sa trajectoire
                self.tracking_objects[self.track_id] = cur
                self.trajectories[self.track_id] = [cur]
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

def plot_trajectories(tracker):
    for track_id, trajectory in tracker.trajectories.items():
        x_vals = [pos[0] for pos in trajectory]
        y_vals = [pos[1] for pos in trajectory]
        plt.plot(x_vals, y_vals, label=f"Véhicule {track_id}")
    
    plt.xlabel("Position X")
    plt.ylabel("Position Y")
    plt.title("Trajectoires des véhicules")
    plt.show()


def main():
    video_in_path = "data/in/video_cut3.mp4"
    video_out_path = "data/out/processed_video_cut3.mp4"
    model = YOLO("traffic_analysis.pt")
    processor = VideoProcessor(video_in_path, video_out_path, model)

    listOfFrames = processor.video_to_frames()
    processed_frames = processor.process_frames(listOfFrames)
    processor.save_processed_video(processed_frames)
    plot_trajectories(processor.tracker)

main()

