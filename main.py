from utils import read_video, save_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner

def main():
    video_frames = read_video('input_videos/08fd33_4.mp4')

    # Initialize tracker
    tracker = Tracker('models/best.pt')
    objects = tracker.get_object_detections(video_frames)

    # Interpolation Ball Positions
    objects["ball"] = tracker.interpolate_ball_position(objects["ball"])

    # Assign Player team
    team_assigner = TeamAssigner()
    team_color = team_assigner.assign_team_color(video_frames[0],
                                    objects['players'][0])
    
    for frame_number, player_track in enumerate(objects['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_number],
                                                 track["bbox"],
                                                 player_id)
            objects["players"][frame_number][player_id]['team'] = team
            objects["players"][frame_number][player_id]['team_color'] = team_assigner.team_colors[team]

    """print("frame 0 players", objects["ball"][0])
    print("----------------")
    print("frame 0 referee", objects["referees"][0])
    print("----------------")
    print("frame 0 player 0, ", objects["players"][0][0])
    print("frame 0 player 1, ", objects["players"][0][1])
    print("frame 0 player 1, ", objects["players"][0][2])"""
    
    tracks = tracker.get_object_tracks(objects)
    tracks["ball"] = objects["ball"]
    tracks["referees"] = objects["referees"]
    # print(tracks["players1"][0])
 
    # Draw outputs
    ## Draw object tracks
    output_video_frames = tracker.draw_anotations(video_frames=video_frames, tracks=tracks, team_color=team_color)


    save_video(output_video_frames, 'output_videos/output_video.avi')


if __name__ == '__main__':
    main()