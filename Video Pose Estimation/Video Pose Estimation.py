
import cv2
import os
import numpy as np
import pandas as pd
import cv2
import os
import numpy as np
import pandas as pd
from moviepy.editor import VideoFileClip, ImageSequenceClip
import mediapipe as mp


video_folder = "/users/home/sheidarighe/Saba/Cubs/Videos"

output_folder = '/users/home/sheidarighe/Saba/Cubs/output/question_3'  

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Folder 'question_3' created inside the 'output' directory.")
else:
    print(f"Folder 'question_3' already exists inside the 'output' directory.")




#########################
# initialization 
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# pose = mp_pose.Pose(
#     static_image_mode=False,          
#     model_complexity=1,               
#     enable_segmentation=False,        
#     min_detection_confidence=0.7,     
#     min_tracking_confidence=0.7,      
#     smooth_landmarks=True             # reduce jitter
# )



############

for video_file in os.listdir(video_folder):

    if video_file.endswith(".mp4"):
        video_path = os.path.join(video_folder, video_file)
        output_video_path = os.path.join(output_folder, f"output_{video_file}")
        csv_path = os.path.join(output_folder, f"{video_file.split('.')[0]}.csv")

        
        clip = VideoFileClip(video_path)
        fps = clip.fps
        width, height = clip.size
        frame_num = 0
        csv_data = []

        
        def frame_processing(frame, frame_num, csv_data):
            
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            
            num_landmarks = len(mp_pose.PoseLandmark)
            joints = [np.nan] * num_landmarks * 2

            if results.pose_landmarks:
                
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                
                for i, landmark in enumerate(results.pose_landmarks.landmark):
                    if i < num_landmarks:
                        joints[i * 2] = landmark.x * width
                        joints[i * 2 + 1] = landmark.y * height

            csv_data.append([frame_num] + joints)
            frame_num += 1

            
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), frame_num, csv_data

        
        def process_clip(clip):
            global frame_num, csv_data
            new_frames = []
            for frame in clip.iter_frames():
                processed_frame, frame_num, csv_data = frame_processing(frame, frame_num, csv_data)
                new_frames.append(processed_frame)
            return new_frames

        processed_frames = process_clip(clip)
        processed_clip = ImageSequenceClip(processed_frames, fps=fps)
        processed_clip.write_videofile(output_video_path, fps=fps)


        if csv_data:
            num_joints = (len(csv_data[0]) - 1) // 2
            column_names = ["frame"] + [f"x_{i}" for i in range(num_joints)] + [f"y_{i}" for i in range(num_joints)]
            df = pd.DataFrame(csv_data, columns=column_names)
            df.to_csv(csv_path, index=False)

