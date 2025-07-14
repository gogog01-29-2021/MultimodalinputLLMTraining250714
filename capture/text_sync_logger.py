
import datetime
import json
import os

def record_text(output_dir=".", session_log_filename="session.jsonl"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    session_log_path = os.path.join(output_dir, session_log_filename)

    print("Enter text. Type 'quit' to stop recording.")

    while True:
        text_input = input("> ")
        if text_input.lower() == 'quit':
            break

        timestamp = datetime.datetime.now().isoformat()
        
        with open(session_log_path, 'a') as f:
            json.dump({'timestamp': timestamp, 'text': text_input}, f)
            f.write('\n')
        print(f"Logged: {text_input}")

if __name__ == '__main__':
    record_text(output_dir='/home/ubuntu/natural_video_llm/capture')


