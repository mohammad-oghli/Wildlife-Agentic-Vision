from pathlib import Path
import subprocess

def load_video_bytes(path: str) -> bytes:
    return Path(path).read_bytes()

def clean_video_for_gemini_code_runner(input_path):
    """
    Removes all metadata and audio, leaving only the raw video stream.
    """
    output_path = "data/cleaned_output/cleaned_wildlife2.mp4"
    
    print(f"Processing video: Removing audio and metadata tracks...")
    
    # -y: Overwrite output if exists
    # -i: Input file
    # -map 0:v: Select ONLY the video stream
    # -an: Explicitly disable audio
    # -c:v copy: Copy the video stream without re-encoding (lightning fast)
    subprocess.run([
            'ffmpeg', '-y', 
            '-i', input_path,
            '-vcodec', 'libx264', 
            '-pix_fmt', 'yuv420p', 
            '-profile:v', 'baseline', 
            '-level', '3.0',
            '-an', # Remove audio entirely to be safe
            output_path
        ], capture_output=True, text=True, check=True)
    
    return output_path
