from pathlib import Path
import subprocess


def extract_audio(video_path: str) -> str:
    """
    Extract audio from a video and save it as a WAV file.
    Returns the path to the generated audio file.
    """

    # Create data/audio folder if it doesn't exist
    audio_dir = Path("data/audio")
    audio_dir.mkdir(parents=True, exist_ok=True)

    video_path = Path(video_path)

    # Output filename will be the same as the video
    audio_path = audio_dir / f"{video_path.stem}.wav"

    command = [
        "ffmpeg",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-y",
        str(audio_path)
    ]

    subprocess.run(command, check=True)

    return str(audio_path)