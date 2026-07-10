from faster_whisper import WhisperModel

model = WhisperModel("large-v3", device="cpu",          # Change to "cuda" if using NVIDIA GPU
    compute_type="int8")



def transcribe(audio_path):
    segments, info = model.transcribe(audio_path)
    transcript=[]
    for segment in segments:
        transcript.append({
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text.strip()
        })

    return transcript