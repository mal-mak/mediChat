from langchain_google_cloud_sql_pg import PostgresVectorStore
from langchain_core.documents.base import Document


def get_relevant_documents(
    query: str, vector_store: PostgresVectorStore, similarity_threshold: float
) -> list[Document]:
    """
    Retrieve relevant documents based on a query using a vector store.

    Args:
        query (str): The search query string.
        vector_store (PostgresVectorStore): An instance of PostgresVectorStore used to retrieve documents.

    Returns:
        list[Document]: A list of documents relevant to the query.
    """

    relevant_docs_scores = vector_store.similarity_search_with_relevance_scores(
        query=query, k=4
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
            Document(page_content: "First doc", metadata: {"title": "Doc 1"}),
            Document(page_content: "Second doc", metadata: {"title": "Doc 1"}
        ]s
        >>> doc_str: str = format_relevant_documents(documents)
        >>> '''
            Source 1: First doc
            -----
            Source 2: Second doc
        '''
    """
    return "\n".join(
        [
            f"SOURCE:{doc.metadata["SOURCE"]}\nQUESTION:\n{doc.page_content}\nANSWER:\n{doc.metadata["answer"]}\nFOCUS AREA:\n{doc.metadata["focus_area"]}\n-----"
            for i, doc in enumerate(documents)
        ]
    )
