
import cv2
import datetime
import os
import json
import time

def record_video(output_dir=".", filename_prefix="webcam_feed", segment_duration=10): # segment_duration in seconds
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(0)  # 0 is the default camera

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: # Handle cases where FPS is not reported correctly
        fps = 30.0 # Default to 30 FPS

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    session_log_path = os.path.join(output_dir, 'session.jsonl')

    print("Press 'q' to quit recording.")

    out = None
    start_time = time.time()
    segment_start_timestamp = datetime.datetime.now().isoformat()
    current_segment_filename = os.path.join(output_dir, f"{filename_prefix}_{segment_start_timestamp.replace(':', '-')}.mp4")
    out = cv2.VideoWriter(current_segment_filename, fourcc, fps, (frame_width, frame_height))

    # Log the start of the video segment
    with open(session_log_path, 'a') as f:
        json.dump({'timestamp': segment_start_timestamp, 'video_file': current_segment_filename, 'event': 'video_segment_start'}, f)
        f.write('\n')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        current_time = time.time()
        if (current_time - start_time) >= segment_duration:
            # End current segment
            if out is not None:
                out.release()
                print(f"Saved video segment: {current_segment_filename}")
                # Log the end of the video segment
                with open(session_log_path, 'a') as f:
                    json.dump({'timestamp': datetime.datetime.now().isoformat(), 'video_file': current_segment_filename, 'event': 'video_segment_end'}, f)
                    f.write('\n')

            # Start new segment
            start_time = current_time
            segment_start_timestamp = datetime.datetime.now().isoformat()
            current_segment_filename = os.path.join(output_dir, f"{filename_prefix}_{segment_start_timestamp.replace(':', '-')}.mp4")
            out = cv2.VideoWriter(current_segment_filename, fourcc, fps, (frame_width, frame_height))
            
            # Log the start of the new video segment
            with open(session_log_path, 'a') as f:
                json.dump({'timestamp': segment_start_timestamp, 'video_file': current_segment_filename, 'event': 'video_segment_start'}, f)
                f.write('\n')

        if out is not None:
            out.write(frame)

        cv2.imshow('Webcam Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    if out is not None:
        out.release()
        print(f"Saved final video segment: {current_segment_filename}")
        # Log the end of the final video segment
        with open(session_log_path, 'a') as f:
            json.dump({'timestamp': datetime.datetime.now().isoformat(), 'video_file': current_segment_filename, 'event': 'video_segment_end'}, f)
            f.write('\n')

    cap.release()
    cv2.destroyAllWindows()
    print(f"Video recording stopped. Segments saved to {output_dir}")

if __name__ == '__main__':
    record_video(output_dir='/home/ubuntu/natural_video_llm/capture', segment_duration=5) # Save 5-second segments


