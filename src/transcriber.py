import os
from faster_whisper import WhisperModel

model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)


def transcribe(audio_path):

    # Validate audio path
    if not isinstance(audio_path, (str, os.PathLike)):
        raise TypeError(
            f"Expected audio file path, got {type(audio_path).__name__}"
        )

    audio_path = os.fspath(audio_path)

    if not os.path.isfile(audio_path):
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    if os.path.getsize(audio_path) == 0:
        raise ValueError(
            f"Audio file is empty: {audio_path}"
        )

    # Transcribe audio
    segments, info = model.transcribe(audio_path)

    transcript = []

    for segment in segments:
        transcript.append({
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text.strip()
        })

    return transcript
