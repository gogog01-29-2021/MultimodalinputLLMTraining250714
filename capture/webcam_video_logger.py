
import cv2
import datetime
import os
import json

def record_video(output_dir='.', filename_prefix='webcam_feed'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(0)  # 0 is the default camera

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # You can use 'XVID' or 'MJPG'
    
    # Prepare for session logging
    session_log_path = os.path.join(output_dir, 'session.jsonl')

    print("Press 'q' to quit recording.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        timestamp = datetime.datetime.now().isoformat()
        output_filename = os.path.join(output_dir, f"{filename_prefix}_{timestamp.replace(':', '-')}.mp4")
        out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
        out.write(frame)

        # Log session info
        with open(session_log_path, 'a') as f:
            json.dump({'timestamp': timestamp, 'video_file': output_filename}, f)
            f.write('\n')

        cv2.imshow('Webcam Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Video recording saved to {output_dir}")

if __name__ == '__main__':
    record_video(output_dir='/home/ubuntu/natural_video_llm/capture')


