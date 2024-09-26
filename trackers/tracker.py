from ultralytics import YOLO
import supervision as cv
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import os
import cv2

class Tracker:
    def __init__(self, model_path: str):
        self.model  = YOLO(model_path)
        self.tracker = cv.ByteTrack()

    def interpolate_ball_position(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing positions
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        
        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions


    def detect_frames(self, frames: list) -> list:
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch

        return detections
    
    def compute_size_bbox(self, bbox: list) -> float:
        x1, y1, x2, y2 = bbox
        size = (x2-x1) * (y2-y1)
        return size
    
    def display_ball_bbox_sizes(self, ball_bbox_sizes: list):
        frames = list(ball_bbox_sizes.keys())
        bbox_sizes = list(ball_bbox_sizes.values())
    
        plt.scatter(frames, bbox_sizes, marker='o')
        plt.xlabel("Frames")
        plt.ylabel("Bbox ball sizes")
        plt.show()

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None, 
                          read_from_stub2=False, stub_path2=None) -> dict:

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks
        
        if read_from_stub2 and stub_path2 is not None and os.path.exists(stub_path2):
            with open(stub_path2, 'rb') as f:
                detections = pickle.load(f)
        else:
            detections = self.detect_frames(frames)
            if stub_path2 is not None:
                with open(stub_path2, 'wb') as f:
                    pickle.dump(detections, f)
        
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        ball_bbox_sizes = {}

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Convert to supervision detection format
            detection_supervision = cv.Detections.from_ultralytics(detection)

            # Convert goalkeeper to player object
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]
            
            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})

            for tracked_object in detection_with_tracks:
                bbox = tracked_object[0].tolist()
                cls_id = tracked_object[3]
                track_id = tracked_object[4]

                if cls_id == cls_names_inv['player']:
                    tracks['players'][frame_num][track_id] = {"bbox": bbox}
                
                if cls_id == cls_names_inv['referee']:
                    tracks['referees'][frame_num][track_id] = {"bbox": bbox}

            for tracked_object in detection_supervision:
                bbox = tracked_object[0].tolist()
                cls_id = tracked_object[3]

                if cls_id == cls_names_inv['ball']:
                    ball_bbox_sizes[frame_num] = self.compute_size_bbox(bbox)
                    tracks['ball'][frame_num][1] = {"bbox": bbox}

        self.display_ball_bbox_sizes(ball_bbox_sizes)
            
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center = (bbox[0]+bbox[2])/2 
        width = bbox[2]-bbox[0]
        
        cv2.ellipse(
            frame,
            center=(int(x_center), y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235, 
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20

        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2

        y1_rect = y2 - rectangle_height//2 + 15
        y2_rect = y2 + rectangle_height//2 + 15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)

            x1_text = x1_rect + 12
            
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect+15)),
                cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x = int((bbox[0]+bbox[2])//2)

        triangle_points = np.array([
            [x,y],
            [x-10, y-20],
            [x+10, y-20]
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2) # contour

        return frame

    def draw_anotations(self, video_frames: list, tracks: dict):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

            # Draw referees
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0,255,255))

            # Draw ball
            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0,255,0))



            output_video_frames.append(frame)

        return output_video_frames