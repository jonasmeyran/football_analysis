import pickle
import pandas as pd
from scipy.stats import mode
import numpy as np
import matplotlib.pyplot as plt

class Corrected_tracker:
    def __init__(self) -> None:
        with open('stubs/reformated_tracks_stubs.pkl', 'rb') as f:
            self.reformated_tracks = pickle.load(f)
        self.nb_frames = max(len(dict_player["num_frames"]) for dict_player in self.reformated_tracks.values())
    
    def delete_empty_tracks(self):
        self.reformated_tracks = {
            track_id: track_info
            for track_id, track_info in self.reformated_tracks.items()
            if len(track_info['num_frames']) > 0
        }

    def get_reformated_tracks(self):
        return self.reformated_tracks
    
    def display_player_team(self, player_id):
        dict_player = self.reformated_tracks[player_id]
        plt.scatter(dict_player['num_frames'], dict_player['team_id'], marker='o')
        plt.xlabel("Frame")
        plt.ylabel("Team")
        plt.title(f"Player id: {player_id}")
        plt.show()

    def reformat(self):
        original_format = []

        for tracking_id, data in self.reformated_tracks.items():
            for frame, team, bbox in zip(data['num_frames'], data['team_id'], data['bbox']):
                while len(original_format) <= frame:
                    original_format.append({})

                if tracking_id not in original_format[frame]:
                    original_format[frame][tracking_id] = {}

                original_format[frame][tracking_id] = {
                    'bbox': bbox,
                    'team': team,
                }

        return original_format
    
    def __compute_distance_beetween_bboxes(self, bbox1, bbox2):
        distance = np.linalg.norm(
            np.array([bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]]) / 2 -
            np.array([bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]]) / 2)
        return distance
    
    def __team_assignment_interpolation(self, dataframe, tracking_id, window_size=24):

        window_size = len(dataframe["frame"]) if len(dataframe["frame"]) < window_size else window_size

        dataframe['corrected_team'] = dataframe['team'].rolling(window=window_size, center=True).apply(lambda x: mode(x)[0] if len(x) > 0 else x)
        dataframe['corrected_team'] = dataframe['corrected_team'].ffill().bfill()
        self.reformated_tracks[tracking_id]['team_id'] = dataframe['corrected_team'].astype(int).tolist()


    def __team_assignement_isolated_player(self, dataframe, tracking_id, distance_threshold=50):
        for i in range(1, len(dataframe)):
            if dataframe.loc[i, 'team'] != dataframe.loc[i - 1, 'team']:
                current_bbox = dataframe.loc[i, 'bbox']
                current_frame = dataframe.loc[i, 'frame']
                
                is_isolated = True
                for other_tracking_id, other_info in self.reformated_tracks.items():
                    if other_tracking_id == tracking_id:
                        continue  
                    
                    if current_frame in other_info['num_frames']:
                        other_index = other_info['num_frames'].index(current_frame)
                        other_bbox = other_info['bbox'][other_index]
                        
                        distance = self.__compute_distance_beetween_bboxes(current_bbox, other_bbox)
                        
                        if distance < distance_threshold:
                            is_isolated = False
                            break
                
                if is_isolated:
                    dataframe.loc[i, 'team'] = dataframe.loc[i - 1, 'team']

        self.reformated_tracks[tracking_id]['team_id'] = dataframe['team'].tolist()

    
    def correct_team_assignment(self):
        for tracking_id, player_info in self.reformated_tracks.items():
            frames, teams, bboxes = player_info.values()
            dataframe = pd.DataFrame({'frame': frames, 'team': teams, 'bbox': bboxes})
            self.__team_assignment_interpolation(dataframe, tracking_id)
        
        for tracking_id, player_info in self.reformated_tracks.items():
            frames, teams, bboxes = player_info.values()
            dataframe = pd.DataFrame({'frame': frames, 'team': teams, 'bbox': bboxes})
            self.__team_assignement_isolated_player(dataframe, tracking_id)

    def __tracking_id_switches(self):
        
        def detect_team_switch():
            switches = {}
            for player_id, player_data in self.reformated_tracks.items():
                previous_team = None
                for frame, team in zip(player_data['num_frames'], player_data['team_id']):
                    if previous_team is not None and team != previous_team:
                        if player_id not in switches: 
                            switches[player_id] = []
                        switches[player_id].append((frame, previous_team, team))
                    previous_team = team
            return switches
        
        def find_switch_pairs(switches, time_window=24):
            pairs = []
            ids = list(switches.keys())
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    id1, id2 = ids[i], ids[j]
                    switches_id1, switches_id2 = switches[id1], switches[id2]

                    for switch1 in switches_id1:
                        for switch2 in switches_id2:
                            if abs(switch1[0] - switch2[0]) <= time_window:
                                pairs.append((id1, id2, switch1[0], switch2[0]))
            return pairs
        
        def filter_switch_pairs_by_proximity(switch_pairs, distance_threshold=50):
            plausible_switch_pairs = []
            for id1, id2, frame1, frame2 in switch_pairs:
                bbox1 = self.reformated_tracks[id1]['bbox'][self.reformated_tracks[id1]['num_frames'].index(frame1)]
                bbox2 = self.reformated_tracks[id2]['bbox'][self.reformated_tracks[id2]['num_frames'].index(frame2)]

                distance = self.__compute_distance_beetween_bboxes(bbox1, bbox2)

                if distance < distance_threshold:
                    plausible_switch_pairs.append((id1, id2, frame1, frame2, distance))

            return plausible_switch_pairs

        switches = detect_team_switch()
        pairs = find_switch_pairs(switches)
        plausible_switch_pairs = filter_switch_pairs_by_proximity(pairs)

        for id1, id2, frame1, frame2, _ in plausible_switch_pairs:
            frames_id1_before, team_id1_before, bbox_id1_before = [], [], []
            frames_id1_after, team_id1_after, bbox_id1_after = [], [], []


            for idx, frame in enumerate(self.reformated_tracks[id1]['num_frames']):
                if frame < frame1:
                    frames_id1_before.append(frame)
                    team_id1_before.append(self.reformated_tracks[id1]['team_id'][idx])
                    bbox_id1_before.append(self.reformated_tracks[id1]['bbox'][idx])
                else:
                    frames_id1_after.append(frame)
                    team_id1_after.append(self.reformated_tracks[id1]['team_id'][idx])
                    bbox_id1_after.append(self.reformated_tracks[id1]['bbox'][idx])

            frames_id2_before, team_id2_before, bbox_id2_before = [], [], []
            frames_id2_after, team_id2_after, bbox_id2_after = [], [], []


            for idx, frame in enumerate(self.reformated_tracks[id2]['num_frames']):
                if frame < frame2:
                    frames_id2_before.append(frame)
                    team_id2_before.append(self.reformated_tracks[id2]['team_id'][idx])
                    bbox_id2_before.append(self.reformated_tracks[id2]['bbox'][idx])
                else:
                    frames_id2_after.append(frame)
                    team_id2_after.append(self.reformated_tracks[id2]['team_id'][idx])
                    bbox_id2_after.append(self.reformated_tracks[id2]['bbox'][idx])


            self.reformated_tracks[id1]['num_frames'] = frames_id1_before + frames_id2_after
            self.reformated_tracks[id1]['team_id'] = team_id1_before + team_id2_after
            self.reformated_tracks[id1]['bbox'] = bbox_id1_before + bbox_id2_after

            self.reformated_tracks[id2]['num_frames'] = frames_id2_before + frames_id1_after
            self.reformated_tracks[id2]['team_id'] = team_id2_before + team_id1_after
            self.reformated_tracks[id2]['bbox'] = bbox_id2_before + bbox_id1_after

    def __tracking_id_merge(self):
        def merge_id(frame_threshold=24, distance_threshold=50):
            pairs, combined = [], []

            for tracking_id, track_data in self.reformated_tracks.items():
                last_frame, team_id, bbox = (values[-1] for values in track_data.values())
                
                if last_frame >= self.nb_frames - frame_threshold - 1: continue

                candidates_id = []

                for next_id, next_data in self.reformated_tracks.items():
                    if next_id == tracking_id or next_data['team_id'][0] != team_id: continue
                    
                    first_frame = next_data['num_frames'][0]
                    if first_frame <= last_frame or first_frame > last_frame + frame_threshold: continue
                        
                    distance = self.__compute_distance_beetween_bboxes(bbox, next_data['bbox'][0])
                    if distance < distance_threshold: candidates_id.append(next_id)
                    
                if len(candidates_id) != 0: pairs.append((tracking_id, min(candidates_id)))

            for pair in pairs:
                found = False
                for group in combined:
                    if pair[0] in group or pair[1] in group:
                        group.update(pair)
                        found = True
                        break
                if not found: combined.append(set(pair))

            return [tuple(sorted(group)) for group in combined]

        combined_pairs = merge_id()
        for group in combined_pairs:
            merged_num_frames, merged_team_id, merged_bbox = [], [], []

            for tracking_id in group:
                merged_num_frames.extend(self.reformated_tracks[tracking_id]['num_frames'])
                merged_team_id.extend(self.reformated_tracks[tracking_id]['team_id'])
                merged_bbox.extend(self.reformated_tracks[tracking_id]['bbox'])

            sorted_data = sorted(zip(merged_num_frames, merged_team_id, merged_bbox), key=lambda x: x[0])
            merged_num_frames, merged_team_id, merged_bbox = zip(*sorted_data)

            base_id = min(group)
            self.reformated_tracks[base_id] = {
                'num_frames': list(merged_num_frames),
                'team_id': list(merged_team_id),
                'bbox': list(merged_bbox)
            }

            for tracking_id in group:
                if tracking_id != base_id: del self.reformated_tracks[tracking_id]


    def correct_tracking_id(self):
        self.__tracking_id_switches()
        self.__tracking_id_merge()
