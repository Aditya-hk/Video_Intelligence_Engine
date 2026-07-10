import streamlit as st
from pathlib import Path
from src.rag import generate
from src.audio import extract_audio
from src.transcriber import transcribe
from src.chunker import chunk_transcript
from src.vectorstore import add_to_vectorstore, retrieve_from_vectorstore
import streamlit as st

api_key = st.secrets["MISTRAL_API_KEY"]

st.set_page_config(
    page_title="Video Intelligence Engine",
    layout="wide"
)

st.markdown("""
<style>
    /* Overall app background */
    .stApp {
        background: linear-gradient(180deg, #0f1117 0%, #161925 100%);
    }

    /* Main title */
    h1 {
        text-align: center;
        background: linear-gradient(90deg, #7F5AF0, #2CB67D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        font-size: 3.6rem !important;
        padding: 1.2rem 0 0.3rem 0;
        letter-spacing: -1px;
    }

    /* Subtitle under title (the st.write right after title) */
    .stApp > header + div .block-container > div:nth-child(1) p:first-of-type {
        text-align: center;
        font-size: 1.15rem;
        color: #9CA3AF !important;
        margin-bottom: 2rem;
    }

    /* Section subheaders */
    h2, h3 {
        color: #E4E4E7 !important;
        border-left: 4px solid #7F5AF0;
        padding-left: 12px;
        margin-top: 2rem !important;
        font-size: 1.5rem !important;
    }

    /* Body text */
    p, .stMarkdown, label {
        color: #CBD5E1 !important;
        font-size: 1.02rem;
    }

    /* Center the file uploader block */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #1A1D29;
        border: 2px dashed #7F5AF0;
        border-radius: 14px;
        padding: 1.5rem;
    }

    [data-testid="stFileUploader"] label {
        font-size: 1.1rem !important;
        font-weight: 600;
        color: #E4E4E7 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #7F5AF0, #2CB67D);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 1.6rem;
        font-weight: 700;
        font-size: 1.02rem;
        transition: 0.25s ease-in-out;
        box-shadow: 0 4px 14px rgba(127, 90, 240, 0.35);
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 22px rgba(127, 90, 240, 0.55);
    }

    /* Text input */
    .stTextInput > div > div > input {
        background-color: #1A1D29;
        color: #E4E4E7;
        border: 1.5px solid #2CB67D;
        border-radius: 10px;
        padding: 0.75rem;
        font-size: 1.05rem;
    }
    .stTextInput > div > div > input:focus {
        border: 1.5px solid #7F5AF0;
        box-shadow: 0 0 0 3px rgba(127, 90, 240, 0.25);
    }

    /* Success / info messages */
    .stAlert {
        border-radius: 12px;
        font-size: 1rem;
        padding: 0.9rem 1.1rem;
    }

    /* Video / Audio players */
    video, audio {
        border-radius: 14px;
        box-shadow: 0 6px 24px rgba(0,0,0,0.45);
        margin: 0.8rem 0;
    }

    /* Dividers */
    hr {
        border-color: #2A2E3F !important;
        margin: 1.2rem 0 !important;
    }

    /* Chunk cards feel */
    .stMarkdown h3 {
        background-color: #1A1D29;
        padding: 0.6rem 1rem;
        border-radius: 10px;
        border-left: 4px solid #2CB67D;
        font-size: 1.2rem !important;
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 10px;
    }
    ::-webkit-scrollbar-track {
        background: #161925;
    }
    ::-webkit-scrollbar-thumb {
        background: #7F5AF0;
        border-radius: 10px;
    }

    /* Sidebar (if used later) */
    section[data-testid="stSidebar"] {
        background-color: #14161F;
    }
</style>


""", unsafe_allow_html=True)

st.title("🎥 Video Intelligence Engine")
st.write("Upload a video to begin processing.")


VIDEO_DIR = Path("data/videos")
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

uploaded_file = st.file_uploader(
    "Choose a video",
    type=["mp4", "mov", "avi", "mkv"]
)


if uploaded_file is not None:

    
    save_path = VIDEO_DIR / uploaded_file.name

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ Video uploaded successfully!")

    st.video(str(save_path))

    # Extract audio
    audio_path = extract_audio(str(save_path))

    st.success("✅ Audio extracted successfully!")

    st.audio(audio_path)

    # Transcribe
    with st.spinner("Transcribing..."):
     transcript = transcribe(audio_path)

    # Chunk transcript
    with st.spinner("Creating chunks..."):
      chunks = chunk_transcript(transcript)


    st.subheader("📄 Transcript")

    for segment in transcript:
        st.write(
            f"[{segment['start']} - {segment['end']}] {segment['text']}"
        )
    st.info("📦 Chunking transcript...")

 
    st.subheader("📦 Chunks")
    
    for i, chunk in enumerate(chunks, start=1):
        st.markdown(f"### Chunk {i}")
        st.write(f"Time: {chunk['start']}s → {chunk['end']}s")
        st.write(chunk["text"])
        st.divider()
    st.success("✅ Chunking complete!")
  
    if st.button("Store Chunks in ChromaDB"):
        add_to_vectorstore(chunks)
        st.success("✅ Chunks stored successfully!")


    st.subheader("💬 Ask Questions")

    query = st.text_input("Ask a question about the uploaded video")

    if query:

        docs = retrieve_from_vectorstore(query)

        st.subheader("🔍 Retrieved Context")

        context = "\n\n".join(
            doc.page_content for doc in docs
        )
        with st.spinner("Generating answer..."):
         answer = generate(context, query)
         st.subheader( "Answer")
         st.write(answer)
