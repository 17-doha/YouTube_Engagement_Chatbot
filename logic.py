import re
import whisper
from youtube_transcript_api import YouTubeTranscriptApi

def is_youtube_link(input_string):
    youtube_regex = (
        r'(https?://)?(www\.)?'
        r'(youtube\.com|youtu\.be)/'
        r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')
    return bool(re.match(youtube_regex, input_string))

def get_video_id(url):
    match = re.match(
        r'(https?://)?(www\.)?'
        r'(youtube\.com|youtu\.be)/'
        r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})', url)
    return match.group(5) if match else None

def generate_transcript(video_url):
    try:
        video_id = get_video_id(video_url)
        if not video_id:
            raise ValueError("Invalid YouTube URL")

        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            text = " ".join([entry['text'] for entry in transcript])
            return text
        except:
            model = whisper.load_model("base")
            result = model.transcribe("video_audio.mp3")  # Placeholder
            return result["text"]
    except Exception as e:
        print(f"Error generating transcript: {e}")
        return None

def process_input(user_input):
    if is_youtube_link(user_input):
        transcript = generate_transcript(user_input)
        return transcript or "Failed to get transcript"
    else:
        return user_input
