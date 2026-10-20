import streamlit as st
import httpx

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# Initialize Groq LLM
@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
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
        print(f"Groq generation error: {type(e).__name__}: {e}")

        # Display useful error information without exposing secrets
        if isinstance(e, httpx.HTTPStatusError):
            st.error(f"Groq API returned HTTP {e.response.status_code}.")
            st.code(e.response.text[:1500], language="text")
        else:
            st.error(
                "Unable to generate an answer using Groq. "
                "Check the application logs for details."
            )

        return None
