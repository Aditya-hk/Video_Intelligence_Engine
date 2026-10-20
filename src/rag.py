import streamlit as st
import httpx

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# Initialize Groq LLM
@st.cache_resource
def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=st.secrets["GROQ_API_KEY"],
        temperature=0.5,
        max_retries=2,
    )


llm = get_llm()


# Define the RAG prompt
prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant.

Answer ONLY using the provided context.

If the answer is not present in the context, say:
"I couldn't find that information in the uploaded video."

Context:
{context}

Question:
{question}
"""
)


# Create the RAG chain
chain = prompt | llm


def generate(context, question):
    try:
        response = chain.invoke({
            "context": context,
            "question": question,
        })
        return response.content

    except Exception as e:
        import traceback

        print("=== GROQ GENERATION ERROR ===")
        print("Error type:", type(e).__name__)
        print("Error message:", str(e))
        traceback.print_exc()

        st.error(f"Groq error: {type(e).__name__}")
        st.code(str(e)[:2000], language="text")

        return None
