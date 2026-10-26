from pathlib import Path
import hashlib
import html
import streamlit as st

from src.rag import generate
from src.audio import extract_audio
from src.transcriber import transcribe
from src.chunker import chunk_transcript
from src.vectorstore import add_to_vectorstore, retrieve_from_vectorstore

st.set_page_config(page_title="Video Intelligence Engine", page_icon="🎬", layout="wide")

VIDEO_DIR = Path("data/videos")
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{--bg:#0b1020;--panel:#111a2e;--border:rgba(159,176,219,.16);--text:#edf2ff;--muted:#a1aec9;--purple:#9a8cff;--cyan:#6ee7e0}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:radial-gradient(ellipse at 12% 0%,rgba(104,87,214,.16),transparent 32%),radial-gradient(ellipse at 92% 12%,rgba(43,183,184,.09),transparent 28%),var(--bg);color:var(--text)}
[data-testid="stHeader"]{background:rgba(11,16,32,.75)}
[data-testid="stMainBlockContainer"]{max-width:1440px;padding-top:2rem;padding-bottom:4rem}
h1,h2,h3{font-family:'Space Grotesk',sans-serif!important;color:var(--text)!important;letter-spacing:-.035em;border:none!important}
p,label,.stMarkdown,[data-testid="stCaptionContainer"]{color:var(--muted)}
.hero{padding:2.2rem 2.4rem;border:1px solid var(--border);border-radius:24px;background:linear-gradient(125deg,rgba(26,36,66,.96),rgba(17,26,46,.94));margin-bottom:1.5rem}
.eyebrow{display:inline-block;padding:.35rem .7rem;border:1px solid rgba(154,140,255,.35);background:rgba(154,140,255,.09);color:#c9c1ff;border-radius:999px;font-size:.76rem;font-weight:700;letter-spacing:.09em;margin-bottom:.9rem}
.hero-title{font-family:'Space Grotesk',sans-serif;color:#f5f7ff;font-size:clamp(2.2rem,4vw,3.6rem);line-height:1.08;font-weight:700;letter-spacing:-.055em;margin:0 0 .85rem}
.gradient-text{background:linear-gradient(90deg,#b2a6ff,#73e6dc);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hero-copy{max-width:760px;color:#b3bfd9;font-size:1.05rem;line-height:1.75;margin-bottom:0}
.section-label{color:#a9b5d2;font-size:.76rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;margin:.6rem 0 .35rem}
.panel{background:linear-gradient(145deg,rgba(21,31,54,.92),rgba(15,24,43,.94));border:1px solid var(--border);border-radius:18px;padding:1.25rem 1.35rem;margin-bottom:1rem}
.metric{background:rgba(18,28,49,.9);border:1px solid var(--border);border-radius:15px;padding:1rem 1.1rem;min-height:95px}
.metric-label{color:#9eacc9;font-size:.78rem;font-weight:600;margin-bottom:.45rem}
.metric-value{color:#f1f4ff;font-family:'Space Grotesk',sans-serif;font-size:1.55rem;font-weight:700}
.answer-card{background:linear-gradient(135deg,rgba(72,63,137,.24),rgba(21,47,64,.36));border:1px solid rgba(154,140,255,.30);border-radius:18px;padding:1.4rem 1.5rem;color:#edf2ff;line-height:1.8;white-space:pre-wrap}
.small-muted{color:#8e9bb7;font-size:.86rem}
div[data-testid="stFileUploader"]{background:rgba(14,22,40,.65);border:1px dashed rgba(154,140,255,.48);border-radius:16px;padding:.7rem}
[data-testid="stFileUploaderDropzone"]{background:rgba(24,34,58,.56);border:0;border-radius:12px}
.stButton>button,.stDownloadButton>button{width:100%;border-radius:10px;border:1px solid rgba(154,140,255,.38);background:linear-gradient(110deg,#8778ef,#5e9fe8);color:white;font-weight:700;min-height:2.8rem;transition:all .18s ease}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:#b7adff;filter:brightness(1.08);transform:translateY(-1px)}
.stTextInput input{background:#0e172a;color:#edf2ff;border:1px solid rgba(159,176,219,.25);border-radius:10px}
div[data-testid="stAlert"]{border-radius:12px}
div[data-testid="stExpander"]{background:rgba(18,28,49,.55);border:1px solid var(--border);border-radius:12px}
hr{border-color:var(--border)}
section[data-testid="stSidebar"]{background:#0d1425;border-right:1px solid var(--border)}
.footer{border-top:1px solid var(--border);padding-top:1rem;margin-top:3rem;color:#8290ad;font-size:.82rem;text-align:center}
</style>
""", unsafe_allow_html=True)

defaults = {
    "processed_key": None, "video_path": None, "audio_path": None,
    "transcript": None, "chunks": None, "stored_key": None,
    "answer": None, "docs": None, "last_query": ""
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

with st.sidebar:
    st.markdown("## 🎬 Video Intelligence")
    st.caption("AI-powered video understanding")
    st.divider()
    st.markdown("### Workflow")
    st.markdown("1. Upload video\n2. Extract audio\n3. Transcribe speech\n4. Build searchable knowledge\n5. Ask questions")
    st.divider()
    st.markdown("### Technology")
    st.markdown("- **Streamlit** · Interface\n- **FFmpeg** · Audio extraction\n- **Faster-Whisper** · Transcription\n- **Sentence Transformers** · Embeddings\n- **ChromaDB** · Vector search\n- **Groq / configured LLM** · Answers")
    st.caption("Upload only videos you have permission to process.")

st.markdown("""
<div class="hero">
  <div class="eyebrow">AI WORKSPACE · VIDEO RAG</div>
  <div class="hero-title">Turn video into<br><span class="gradient-text">searchable knowledge.</span></div>
  <p class="hero-copy">Upload a video, generate a timestamped transcript, index its content, and ask natural-language questions grounded in the material.</p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.15, .85], gap="large")
with left:
    st.markdown('<div class="section-label">01 / Source material</div>', unsafe_allow_html=True)
    st.subheader("Upload a video")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv"], help="Supported formats: MP4, MOV, AVI, MKV.")
    st.caption("For best results, use clear audio and stay within your hosting provider's upload limits.")

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_key = hashlib.sha256(file_bytes).hexdigest()
        suffix = Path(uploaded_file.name).suffix.lower() or ".mp4"
        save_path = VIDEO_DIR / f"{file_key[:16]}{suffix}"

        if st.session_state.processed_key != file_key:
            st.session_state.processed_key = file_key
            st.session_state.video_path = None
            st.session_state.audio_path = None
            st.session_state.transcript = None
            st.session_state.chunks = None
            st.session_state.stored_key = None
            st.session_state.answer = None
            st.session_state.docs = None
            st.session_state.last_query = ""
            save_path.write_bytes(file_bytes)
            st.session_state.video_path = str(save_path)
        elif not save_path.exists():
            save_path.write_bytes(file_bytes)
            st.session_state.video_path = str(save_path)

        st.video(file_bytes)
        st.markdown(f'<div class="small-muted">Selected file: {html.escape(uploaded_file.name)} · {len(file_bytes)/(1024*1024):.1f} MB</div>', unsafe_allow_html=True)
        if st.button("⚡ Process video", key="process_video"):
            try:
                progress = st.progress(0, text="Preparing video…")
                with st.spinner("Extracting audio with FFmpeg…"):
                    audio_path = extract_audio(st.session_state.video_path)
                st.session_state.audio_path = str(audio_path)
                progress.progress(25, text="Audio extracted · transcribing speech…")
                with st.spinner("Transcribing audio with Faster-Whisper…"):
                    transcript = transcribe(st.session_state.audio_path)
                st.session_state.transcript = transcript
                progress.progress(65, text="Transcript ready · creating semantic chunks…")
                with st.spinner("Chunking transcript…"):
                    chunks = chunk_transcript(transcript)
                st.session_state.chunks = chunks
                progress.progress(100, text="Video processing complete")
                st.success("Video processed. Review the transcript and index its chunks below.")
            except Exception as exc:
                st.error(f"Processing failed: {exc}")
                st.caption("Check FFmpeg, model availability, and server logs.")
    else:
        st.markdown('<div class="panel"><h3>Start with a video</h3><p>Upload a lecture, interview, presentation, or other video with speech. The application prepares spoken content for semantic search.</p></div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-label">02 / Pipeline overview</div>', unsafe_allow_html=True)
    st.subheader("Processing pipeline")
    stages = [
        ("01", "Audio extraction", "Extract audio from the video.", st.session_state.audio_path is not None),
        ("02", "Speech transcription", "Convert speech into timestamped text.", st.session_state.transcript is not None),
        ("03", "Semantic chunking", "Split transcript into searchable segments.", st.session_state.chunks is not None),
        ("04", "Vector indexing", "Store chunks for similarity retrieval.", st.session_state.stored_key == st.session_state.processed_key and st.session_state.processed_key is not None),
    ]
    for number, title, description, complete in stages:
        status = "✓ Complete" if complete else "Pending"
        color = "#7ce0b1" if complete else "#8e9bb7"
        st.markdown(f'<div class="panel" style="padding:.9rem 1rem;margin-bottom:.7rem"><div style="display:flex;justify-content:space-between;gap:12px;align-items:center"><strong style="color:#edf2ff">{number} &nbsp; {title}</strong><span style="color:{color};font-size:.78rem;font-weight:700">{status}</span></div><div class="small-muted" style="margin-top:.35rem">{description}</div></div>', unsafe_allow_html=True)

if st.session_state.transcript is not None:
    transcript = st.session_state.transcript
    chunks = st.session_state.chunks or []
    st.divider()
    st.markdown('<div class="section-label">03 / Generated content</div>', unsafe_allow_html=True)
    st.subheader("Transcript & semantic chunks")
    word_count = sum(len(str(segment.get("text", "")).split()) for segment in transcript)
    m1, m2, m3 = st.columns(3)
    for col, label, value in [
        (m1, "TRANSCRIPT SEGMENTS", len(transcript)),
        (m2, "SEMANTIC CHUNKS", len(chunks)),
        (m3, "TRANSCRIBED WORDS", f"{word_count:,}"),
    ]:
        with col:
            st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)

    tab_transcript, tab_chunks = st.tabs(["📄 Transcript", "🧩 Chunks"])
    with tab_transcript:
        search_text = st.text_input("Search transcript", placeholder="Find a word or phrase…")
        filtered = [s for s in transcript if not search_text or search_text.lower() in str(s.get("text", "")).lower()]
        if not filtered:
            st.info("No transcript segments match your search.")
        for idx, segment in enumerate(filtered, start=1):
            with st.expander(f'{segment.get("start", "?")}s – {segment.get("end", "?")}s · Segment {idx}'):
                st.write(segment.get("text", ""))
        transcript_text = "\n".join(f'[{s.get("start", "?")} - {s.get("end", "?")}] {s.get("text", "")}' for s in transcript)
        st.download_button("Download transcript (.txt)", transcript_text, file_name="video_transcript.txt", mime="text/plain")
    with tab_chunks:
        if not chunks:
            st.info("No chunks were generated.")
        for idx, chunk in enumerate(chunks, start=1):
            with st.expander(f'Chunk {idx:02d} · {chunk.get("start", "?")}s–{chunk.get("end", "?")}s'):
                st.write(chunk.get("text", ""))
        if st.button("Save chunks to semantic index", key="store_chunks"):
            try:
                with st.spinner("Adding chunks to ChromaDB…"):
                    add_to_vectorstore(chunks)
                st.session_state.stored_key = st.session_state.processed_key
                st.session_state.answer = None
                st.success("Chunks added to the vector store.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not store chunks: {exc}")

if st.session_state.stored_key == st.session_state.processed_key and st.session_state.processed_key is not None:
    st.divider()
    st.markdown('<div class="section-label">04 / Ask your video</div>', unsafe_allow_html=True)
    st.subheader("Chat with your content")
    st.caption("Answers are generated using relevant transcript context.")
    with st.form("question_form", clear_on_submit=False):
        query = st.text_input("Your question", placeholder="e.g. What are the main ideas discussed in this video?")
        submitted = st.form_submit_button("✦ Generate answer")
    if submitted:
        if not query.strip():
            st.warning("Enter a question first.")
        else:
            try:
                with st.spinner("Searching transcript and generating an answer…"):
                    docs = retrieve_from_vectorstore(query.strip())
                    st.session_state.docs = docs
                    context = "\n\n".join(doc.page_content for doc in docs)
                    st.session_state.answer = generate(context, query.strip())
                    st.session_state.last_query = query.strip()
            except Exception as exc:
                st.error(f"Could not generate an answer: {exc}")
    if st.session_state.answer:
        st.markdown("#### Answer")
        st.markdown(f'<div class="answer-card">{html.escape(str(st.session_state.answer))}</div>', unsafe_allow_html=True)
        docs = st.session_state.docs or []
        if docs:
            with st.expander(f"View retrieved context ({len(docs)} result(s))"):
                for idx, doc in enumerate(docs, start=1):
                    st.markdown(f"**Source {idx}**")
                    st.write(doc.page_content)
                    metadata = getattr(doc, "metadata", {}) or {}
                    if metadata:
                        st.caption(" · ".join(f"{k}: {v}" for k, v in metadata.items()))
                    if idx < len(docs):
                        st.divider()

st.markdown('<div class="footer">VIDEO INTELLIGENCE ENGINE · Speech recognition · Vector search · Retrieval-augmented generation</div>', unsafe_allow_html=True)
