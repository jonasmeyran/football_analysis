from utils import read_video, save_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner

def main():
    video_frames = read_video('input_videos/08fd33_4.mp4')

    # Initialize tracker
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl',
                                       read_from_stub2=True, stub_path2="stubs/detection_stubs.pkl")

    # Interpolation Ball Positions
    tracks["ball"] = tracker.interpolate_ball_position(tracks["ball"])

    # Assign Player team
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0],
                                    tracks['players'][0])
    for frame_number, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_number],
                                                 track["bbox"],
                                                 player_id)
            tracks["players"][frame_number][player_id]['team'] = team
            tracks["players"][frame_number][player_id]['team_color'] = team_assigner.team_colors[team]

    # Draw outputs
    ## Draw object tracks
    output_video_frames = tracker.draw_anotations(video_frames=video_frames, tracks=tracks)


    save_video(output_video_frames, 'output_videos/output_video.avi')


if __name__ == '__main__':
    main()