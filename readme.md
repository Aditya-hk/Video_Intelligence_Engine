# 🎥 Video Intelligence Engine

An end-to-end AI-powered application that enables users to upload videos and ask natural language questions about their content. Instead of manually watching long videos, the application extracts audio, generates timestamped transcripts, builds a semantic vector index, and retrieves the most relevant context to generate accurate, context-aware answers using Retrieval-Augmented Generation (RAG).

---

## 🚀 Features

- 📹 Upload video files through a simple Streamlit interface
- 🎧 Automatic audio extraction using FFmpeg
- 📝 Fast and accurate timestamped transcription with Faster-Whisper
- ✂️ Semantic transcript chunking for improved retrieval quality
- 🔍 Vector embeddings using Hugging Face Sentence Transformers
- 🗄️ ChromaDB vector database for efficient semantic search
- 🤖 Context-aware question answering using Mistral AI
- 🧩 Modular architecture with separate processing pipelines
- ⚡ Interactive web interface built with Streamlit

---

## 🏗️ System Architecture

```text
                +-------------------+
                |   Upload Video    |
                +---------+---------+
                          |
                          v
                +-------------------+
                | Audio Extraction  |
                |     (FFmpeg)      |
                +---------+---------+
                          |
                          v
                +-------------------+
                | Faster Whisper    |
                | Transcription     |
                +---------+---------+
                          |
                          v
                +-------------------+
                | Semantic Chunking |
                +---------+---------+
                          |
                          v
                +-------------------+
                | HuggingFace       |
                | Embeddings        |
                +---------+---------+
                          |
                          v
                +-------------------+
                | ChromaDB          |
                | Vector Store      |
                +---------+---------+
                          |
                          v
                +-------------------+
                | Retrieve Context  |
                +---------+---------+
                          |
                          v
                +-------------------+
                | Mistral AI        |
                | Answer Generator  |
                +---------+---------+
                          |
                          v
                +-------------------+
                | Final Response    |
                +-------------------+
```

---

## 📂 Project Structure

```text
Video_Intelligence_Engine/
│
├── app.py
├── requirements.txt
├── README.md
│
├── pipelines/
│   ├── audio_extractor.py
│   ├── transcriber.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vectorstore.py
│   └── rag_pipeline.py
│
├── utils/
│   ├── helpers.py
│   └── config.py
│
├── uploads/
├── transcripts/
├── chroma_db/
│
└── assets/
```

> Folder names may vary depending on your implementation.

---

# 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| Frontend | Streamlit |
| LLM Framework | LangChain |
| Speech-to-Text | Faster-Whisper |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Database | ChromaDB |
| LLM | Mistral AI |
| Audio Processing | FFmpeg |

---

# ⚙️ Installation

## 1. Clone Repository

```bash
git clone https://github.com/Aditya-hk/Video_Intelligence_Engine.git

cd Video_Intelligence_Engine
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Install FFmpeg

### Windows

Download FFmpeg and add it to your system PATH.

### macOS

```bash
brew install ffmpeg
```

### Ubuntu

```bash
sudo apt install ffmpeg
```

---

## 5. Configure Environment Variables

Create a `.env` file.

```env
MISTRAL_API_KEY=your_api_key_here
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

# 💡 How It Works

### Step 1

Upload a video.

↓

### Step 2

Extract audio using FFmpeg.

↓

### Step 3

Generate timestamped transcript using Faster-Whisper.

↓

### Step 4

Split transcript into semantic chunks.

↓

### Step 5

Generate vector embeddings.

↓

### Step 6

Store embeddings inside ChromaDB.

↓

### Step 7

Retrieve relevant chunks based on user query.

↓

### Step 8

Send retrieved context to Mistral AI.

↓

### Step 9

Generate an accurate answer grounded in the transcript.

---

# 📸 Demo

## Upload Video

> Upload your video file using the Streamlit interface.

---

## Ask Questions

Example queries:

```
Summarize this lecture.

What is the speaker's main argument?

Explain the machine learning algorithm discussed.

What are the key takeaways?

Give me a summary of the first half.
```

---

# 🎯 Learning Outcomes

This project helped me gain hands-on experience with:

- Retrieval-Augmented Generation (RAG)
- Large Language Model integration
- Speech-to-text pipelines
- Semantic search
- Vector databases
- Embedding models
- Streamlit application development
- Modular software architecture
- End-to-end AI workflow design

---

# 🚀 Future Improvements

- [ ] Speaker Diarization
- [ ] OCR for on-screen text
- [ ] Timestamp-aware answers
- [ ] Multimodal Retrieval
- [ ] Hybrid Search (Keyword + Vector)
- [ ] Streaming transcription
- [ ] Support for YouTube URLs
- [ ] Conversation memory
- [ ] Citation highlighting
- [ ] Docker support
- [ ] GPU inference optimization

---

# 📊 Workflow Overview

```text
Video
   │
   ▼
Audio Extraction
   │
   ▼
Whisper Transcription
   │
   ▼
Semantic Chunking
   │
   ▼
Embeddings
   │
   ▼
ChromaDB
   │
   ▼
Retriever
   │
   ▼
Mistral AI
   │
   ▼
Answer
```

---

# 🤝 Contributing

Contributions are welcome!

Feel free to:

- Fork the repository
- Create a feature branch
- Open an issue
- Submit a pull request

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Aditya Kumar**

B.Tech Information Technology  
Indian Institute of Information Technology Bhopal

GitHub: https://github.com/Aditya-hk

LinkedIn: https://www.linkedin.com/in/aditya-hk/

---

⭐ If you found this project useful, consider giving it a star!
