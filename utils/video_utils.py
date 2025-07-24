import cv2
import subprocess
import os

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(output_video_frames, output_video_path):
    # First save with OpenCV
    temp_output = output_video_path + '.temp.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(temp_output, fourcc, 24, 
                         (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    
    for frame in output_video_frames:
        out.write(frame)
    out.release()
    
    # Convert using FFmpeg with high quality settings
    try:
        ffmpeg_cmd = [
            'ffmpeg', '-y',  # Overwrite output file if it exists
            '-i', temp_output,  # Input file
            '-c:v', 'libx264',  # Video codec
            '-preset', 'medium',  # Encoding speed preset
            '-crf', '23',  # Quality (lower = better, 18-28 is good)
            '-pix_fmt', 'yuv420p',  # Pixel format for maximum compatibility
            output_video_path  # Output file
        ]
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        
        # Remove temporary file
        if os.path.exists(temp_output):
            os.remove(temp_output)
            
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        # Fallback to the temporary file if FFmpeg fails
        if os.path.exists(temp_output):
            os.rename(temp_output, output_video_path)
    except Exception as e:
        print(f"Error during video conversion: {str(e)}")
        # Fallback to the temporary file
        if os.path.exists(temp_output):
            os.rename(temp_output, output_video_path)
