
import json
import os
from datetime import datetime, timedelta

def align_text_video(session_log_path, output_dir=".", alignment_filename="aligned_data.jsonl", time_window_seconds=5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    aligned_data_path = os.path.join(output_dir, alignment_filename)

    video_segments = []
    text_entries = []

    current_video_segment = None

    with open(session_log_path, \'r\') as f:
        for line in f:
            entry = json.loads(line.strip())
            entry_time = datetime.fromisoformat(entry[\'timestamp\'])

            if \'event\' in entry and entry[\'event\'] == \'video_segment_start\':
                if current_video_segment is not None: # Close previous segment if open
                    current_video_segment[\'end_time\'] = entry_time # Use start of new segment as end of old
                    video_segments.append(current_video_segment)
                current_video_segment = {
                    \'start_time\': entry_time,
                    \'video_file\': entry[\'video_file\'],
                    \'end_time\': None # Will be filled by next segment start or end of log
                }
            elif \'event\' in entry and entry[\'event\'] == \'video_segment_end\':
                if current_video_segment is not None and current_video_segment[\'video_file\'] == entry[\'video_file\']:
                    current_video_segment[\'end_time\'] = entry_time
                    video_segments.append(current_video_segment)
                    current_video_segment = None
            elif \'text\' in entry:
                text_entries.append({\'time\': entry_time, \'text\': entry[\'text\']})

    # Add any remaining open video segment
    if current_video_segment is not None and current_video_segment[\'end_time\'] is None:
        current_video_segment[\'end_time\'] = datetime.now() # Assume end at current time if not explicitly logged
        video_segments.append(current_video_segment)

    # Sort entries by time to ensure correct alignment
    video_segments.sort(key=lambda x: x[\'start_time\'])
    text_entries.sort(key=lambda x: x[\'time\'])

    with open(aligned_data_path, \'w\') as outfile:
        for text_entry in text_entries:
            text_time = text_entry[\'time\']
            associated_video_files = []
            
            # Find video segments that overlap with the text entry's time window
            for video_segment in video_segments:
                segment_start = video_segment[\'start_time\']
                segment_end = video_segment[\'end_time\']
                
                # Check for overlap: text_time within segment or segment overlaps text_time window
                if segment_end and (
                    (text_time >= segment_start and text_time <= segment_end) or
                    (text_time - timedelta(seconds=time_window_seconds) <= segment_end and \
                     text_time + timedelta(seconds=time_window_seconds) >= segment_start)
                ):
                    associated_video_files.append(video_segment[\'video_file\'])
            
            if associated_video_files:
                # Ensure unique video files
                associated_video_files = list(set(associated_video_files))
                json.dump({
                    \'text_timestamp\': text_entry[\'time\'].isoformat(),
                    \'text\': text_entry[\'text\'],
                    \'associated_video_files\': associated_video_files
                }, outfile)
                outfile.write(\'\\n\')

    print(f"Aligned data saved to {aligned_data_path}")

if __name__ == \'__main__\':
    session_log_file = \'/home/ubuntu/natural_video_llm/capture/session.jsonl\'
    if os.path.exists(session_log_file):
        align_text_video(session_log_file, output_dir=\'/home/ubuntu/natural_video_llm/preprocessing\')
    else:
        print(f"Error: {session_log_file} not found. Please run webcam_video_logger.py and text_sync_logger.py first.")


