def chunk_transcript(transcript, max_chars=5):
    """
    Groups transcript segments into larger semantic chunks.
    """

    chunks = []

    current_text = ""
    start_time = None
    end_time = None

    for segment in transcript:

        if start_time is None:
            start_time = segment["start"]

        current_text += " " + segment["text"]
        end_time = segment["end"]

        if len(current_text) >= max_chars:

            chunks.append({
                "start": start_time,
                "end": end_time,
                "text": current_text.strip()
            })

            current_text = ""
            start_time = None
            end_time = None

    # Remaining text
    if current_text:

        chunks.append({
            "start": start_time,
            "end": end_time,
            "text": current_text.strip()
        })

    return chunks