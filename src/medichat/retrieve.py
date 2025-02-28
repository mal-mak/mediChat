from langchain_google_cloud_sql_pg import PostgresVectorStore
from langchain_core.documents.base import Document


def get_relevant_documents(
    query: str,
    vector_store: PostgresVectorStore,
    similarity_threshold: float,
    max_sources: float,
) -> list[Document]:
    """
    Retrieve relevant documents based on a query using a vector store.

    Args:
        query (str): The search query string.
        vector_store (PostgresVectorStore): An instance of PostgresVectorStore used to retrieve documents.
        similarity_threshold (float): The similarity threshold to use for the search.
        max_sources (int): The maximum number of sources to return.

    Returns:
        list[Document]: A list of documents relevant to the query.
    """

    relevant_docs_scores = vector_store.similarity_search_with_relevance_scores(
        query=query, k=max_sources
    )
    for doc, score in relevant_docs_scores:
        doc.metadata["score"] = score
    relevant_docs = [doc for doc, _ in relevant_docs_scores]

    return relevant_docs


def format_relevant_documents(documents: list[Document]) -> str:
    """
    Format relevant documents into a str.

    Args:
        documents (list[Document]): A list of relevant documents.

    Returns:
        list[dict]: A list of dictionaries containing the relevant documents.

    Example:
        >>> documents = [
            Document(page_content: "question1", metadata: {"source": "source1", "answer": "answer1", "focus_area": "focus_area1"}),
            Document(page_content: "question2", metadata: {"source": "source2", "answer": "answer2", "focus_area": "focus_area2"})
        ]
        >>> doc_str: str = format_relevant_documents(documents)
        >>> '''
            SOURCE: source1
            QUESTION:
            question1
            ANSWER:
            answer1
            FOCUS AREA:
            focus_area1
            -----
            SOURCE: source2
            ...
        '''
    """
    return "\n".join(
        [
            f"SOURCE:{doc.metadata["source"]}\nQUESTION:\n{doc.page_content}\nANSWER:\n{doc.metadata["answer"]}\nFOCUS AREA:\n{doc.metadata["focus_area"]}\n-----"
            for i, doc in enumerate(documents)
        ]
    )
