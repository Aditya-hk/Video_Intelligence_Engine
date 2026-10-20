import streamlit as st
import httpx

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate


# Initialize Mistral LLM
llm = ChatMistralAI(
    model="mistral-medium-latest",
    api_key=st.secrets["MISTRAL_API_KEY"],
    temperature=0.5,
)

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

    except httpx.HTTPStatusError as e:
        status_code = e.response.status_code
        error_body = e.response.text[:1500]

        # Record the details in Streamlit Cloud logs
        print(f"Mistral HTTP status: {status_code}")
        print(f"Mistral API error: {error_body}")

        # Show a safe message to the user
        if status_code == 401:
            st.error("Mistral authentication failed. Check your API key.")
        elif status_code == 402:
            st.error("Mistral billing or credit issue. Check your account.")
        elif status_code == 403:
            st.error("Mistral access denied. Check your permissions and model access.")
        elif status_code == 404:
            st.error("Mistral model not found. Check the configured model name.")
        elif status_code == 429:
            st.error("Mistral rate limit or usage quota reached. Try again later.")
        else:
            st.error(
                f"Mistral API request failed (HTTP {status_code}). "
                "Check the application logs for details."
            )

        return None

    except Exception as e:
        print(f"RAG generation error: {type(e).__name__}: {e}")
        st.error("An unexpected error occurred while generating the answer.")
        return None
