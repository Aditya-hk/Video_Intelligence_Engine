from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
import os
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate

llm=ChatMistralAI(
    model_name="mistral-medium-latest",
    temperature=0.5,)

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

chain= prompt | llm

def generate(context,question):
    response=chain.invoke({
        "context": context,
        "question": question
    })
    return response.content


