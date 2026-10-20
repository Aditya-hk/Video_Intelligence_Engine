import time

import streamlit as st
import httpx

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate


# Initialize Mistral LLM
@st.cache_resource
def get_llm():
    return ChatMistralAI(
        model="mistral-medium-latest",
        api_key=st.secrets["MISTRAL_API_KEY"],
        temperature=0.5,
        max_retries=0,
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


# Retry configuration
MAX_ATTEMPTS = 3
INITIAL_RETRY_DELAY = 2


def generate(context, question):
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = chain.invoke(
                {
                    "context": context,
                    "question": question,
                }
            )

            return response.content

        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            error_body = e.response.text[:1500]

            # Display the actual API error
            print(f"Mistral HTTP status: {status_code}")
            print(f"Mistral API error: {error_body}")

            if status_code == 429:
                # Retry only when the API reports rate limiting.
                if attempt < MAX_ATTEMPTS - 1:
                    retry_after = e.response.headers.get("Retry-After")

                    try:
                        delay = (
                            max(1.0, float(retry_after))
                            if retry_after
                            else INITIAL_RETRY_DELAY * (2 ** attempt)
                        )
                    except ValueError:
                        delay = INITIAL_RETRY_DELAY * (2 ** attempt)

                    st.warning(
                        f"Mistral rate limit reached. "
                        f"Retrying in {delay:g} seconds "
                        f"({attempt + 1}/{MAX_ATTEMPTS - 1})..."
                    )

                    time.sleep(delay)
                    continue

                st.error(
                    "Mistral rate limit persists after retries. "
                    "Wait before trying again and check your API usage."
                )
                st.code(error_body, language="json")
                return None

            elif status_code == 401:
                st.error(
                    "Mistral authentication failed. "
                    "Check MISTRAL_API_KEY in Streamlit Secrets."
                )

            elif status_code == 402:
                st.error(
                    "Mistral billing or credit issue. "
                    "Check your account."
                )

            elif status_code == 403:
                st.error(
                    "Mistral access denied. "
                    "Check your account permissions and model access."
                )

            elif status_code == 404:
                st.error(
                    "Mistral model not found. "
                    "Check the configured model name."
                )

            else:
                st.error(
                    f"Mistral API request failed (HTTP {status_code})."
                )

            st.code(error_body, language="json")
            return None

        except Exception as e:
            print(f"RAG generation error: {type(e).__name__}: {e}")
            st.error(
                f"Answer generation failed: {type(e).__name__}. "
                "Check the application logs."
            )
            return None

    return None
