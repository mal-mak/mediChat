"""Malek's RAG Medical Chatbot API"""

from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from google.cloud import storage
from dotenv import load_dotenv
from medichat.ingest import (
    create_cloud_sql_database_connection,
    get_embeddings,
    get_vector_store,
    list_files_in_bucket,
)
from medichat.retrieve import get_relevant_documents, format_relevant_documents
from medichat.config import TABLE_NAME, BUCKET_NAME

load_dotenv()

app = FastAPI()
client = storage.Client()

# Initialize once and reuse
ENGINE = create_cloud_sql_database_connection()
EMBEDDING = get_embeddings()


class DocumentResponse(BaseModel):
    page_content: str
    metadata: dict


class UserInput(BaseModel):
    """
    UserInput is a data model representing user input.

    Attributes:
        question (str): The question of the user.
        temperature (float): The temperature of the user.
        language (str): The language preference of the user.
        documents (List[DocumentResponse]): Retrieved documents for context.
    """

    question: str
    temperature: float
    language: str
    similarity_threshold: float
    max_sources: float
    documents: List[DocumentResponse]
    previous_context: List[dict]


@app.post("/get_files_names")
def get_files_names():
    """
    Retrieve the list of available files in the configured Google Cloud Storage bucket.

    Returns:
        dict: A dictionary containing the list of file names under the 'files' key.
    """
    bucket = client.get_bucket(BUCKET_NAME)
    files = list_files_in_bucket(client, bucket)
    return {"files": files}


@app.post("/get_sources", response_model=List[DocumentResponse])
def get_sources(user_input: UserInput) -> List[DocumentResponse]:
    """
    Retrieve relevant source documents based on the user's question.

    Args:
        user_input (UserInput): User input containing the question and retrieval parameters.

    Returns:
        List[DocumentResponse]: A list of relevant documents with their content and metadata.
        Returns empty list if no relevant documents are found.
    """
    vector_store = get_vector_store(ENGINE, TABLE_NAME, EMBEDDING)
    relevants_docs = get_relevant_documents(
        f"Retrieve information related to: {user_input.question}",
        vector_store,
        user_input.similarity_threshold,
        user_input.max_sources,
    )

    if not relevants_docs:
        return []

    # else
    return [
        DocumentResponse(page_content=doc.page_content, metadata=doc.metadata)
        for doc in relevants_docs
    ]


@app.post("/answer")
def answer(user_input: UserInput):
    """
    Generate an answer to a medical question using RAG methodology.

    This function:
    1. Uses the provided source documents as context
    2. Considers previous conversation context
    3. Generates a response using Google's Generative AI
    4. Handles multi-language support

    Args:
        user_input (UserInput): Object containing:
            - question: The medical query
            - temperature: Controls response randomness
            - language: Desired response language
            - documents: Retrieved context documents
            - previous_context: Previous conversation history
            - similarity_threshold: Minimum similarity score for document retrieval
            - max_sources: Maximum number of sources to consider

    Returns:
        dict: Contains the generated answer under the 'message' key
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=user_input.temperature,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    prompt = ChatPromptTemplate.from_messages(
        messages=[
            (
                "system",
                """DOCUMENT:
                {formatted_docs}

                PREVIOUS CONTEXT:
                {previous_context}

                LAST DISCUSSED ENTITY:
                {last_entity}

                INSTRUCTIONS:
                0. You are a knowledgeable medical professional.
                1. You answer questions using ONLY the provided DOCUMENT.
                2. If the QUESTION is in an other language, translate it first to english.
                3. In the DOCUMENT, you can find the ANSWER to the question, the SOURCE of the ANSWER, as well as the FOCUS AREA.
                4. Answer in {language} the QUESTION using the provided DOCUMENT text above.
                5. Keep your answer grounded in the facts from the DOCUMENT only.
                6. Be somewhat concise but retain all relevant information and details.
                7. If the question refers to "it" or any other ambiguous term, refer to the LAST DISCUSSED ENTITY unless further clarification is provided in the QUESTION.
                8. Use the PREVIOUS CONTEXT only if it provides additional clarity or information that directly supports answering the QUESTION.

                QUESTION:
                {question}
                """,
            ),
            ("human", "The query is: {question}"),
        ]
    )

    chain = prompt | llm
    answer = chain.invoke(
        {
            "language": user_input.language,
            "question": user_input.question,
            "formatted_docs": format_relevant_documents(user_input.documents),
            "previous_context": user_input.previous_context,
            "last_entity": user_input.previous_context[-3:-1],
        }
    ).content
    return {"message": answer}
