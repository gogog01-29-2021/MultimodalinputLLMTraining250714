
import json
import os
from datetime import datetime, timedelta

def align_text_video(session_log_path, output_dir=".", alignment_filename="aligned_data.jsonl", time_window_seconds=5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    aligned_data_path = os.path.join(output_dir, alignment_filename)

    video_entries = []
    text_entries = []

    with open(session_log_path, \'r\') as f:
        for line in f:
            entry = json.loads(line.strip())
            entry_time = datetime.fromisoformat(entry[\'timestamp\'])
            if \'video_file\' in entry:
                video_entries.append({\'time\': entry_time, \'file\': entry[\'video_file\]})
            elif \'text\' in entry:
                text_entries.append({\'time\': entry_time, \'text\': entry[\'text\]})

    # Sort entries by time to ensure correct alignment
    video_entries.sort(key=lambda x: x[\'time\'])
    text_entries.sort(key=lambda x: x[\'time\'])

    with open(aligned_data_path, \'w\') as outfile:
        for text_entry in text_entries:
            text_time = text_entry[\'time\']
            associated_videos = []
            
            # Find video frames within the time window around the text entry
            for video_entry in video_entries:
                video_time = video_entry[\'time\']
                if abs((text_time - video_time).total_seconds()) <= time_window_seconds:
                    associated_videos.append(video_entry[\'file\'])
            
            if associated_videos:
                json.dump({
                    \'text_timestamp\': text_entry[\'time\'].isoformat(),
                    \'text\': text_entry[\'text\'],
                    \'associated_video_files\': associated_videos
                }, outfile)
                outfile.write(\'\n\')

    print(f"Aligned data saved to {aligned_data_path}")

if __name__ == \'__main__\':
    # Example usage:
    # Assuming session.jsonl is in the capture directory
    session_log_file = \'/home/ubuntu/natural_video_llm/capture/session.jsonl\'
    if os.path.exists(session_log_file):
        align_text_video(session_log_file, output_dir=\'/home/ubuntu/natural_video_llm/preprocessing\')
    else:
        print(f"Error: {session_log_file} not found. Please run webcam_video_logger.py and text_sync_logger.py first.")


