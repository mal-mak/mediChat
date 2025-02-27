import os
from dotenv import load_dotenv
from sqlalchemy.exc import ProgrammingError
from google.cloud import storage
from google.cloud.storage.bucket import Bucket
from langchain_google_cloud_sql_pg import PostgresEngine, PostgresVectorStore
from langchain_google_vertexai import VertexAIEmbeddings
from google.cloud.exceptions import NotFound
from google.cloud.exceptions import GoogleCloudError

# Non sensitive information goes in config
from config import PROJECT_ID, REGION, INSTANCE, DATABASE, DB_USER

load_dotenv()
# Sensitive information goes in .env
DB_PASSWORD = os.environ["DB_PASSWORD"]

DOWNLOADED_LOCAL_DIRECTORY = "./downloaded_files"


def list_files_in_bucket(
    client: storage.Client, bucket_name: Bucket, directory_name: str = "data/"
) -> list[str]:
    """
    List all the files in the specified Google Cloud Storage bucket.

    Args:
        client (storage.Client): The Google Cloud Storage client.
        bucket_name (Bucket): The name of the bucket to list files from.
        directory_name (str, optional): The directory within the bucket to list files from. Defaults to 'data/'.

    Returns:
        list[str]: A list of file names in the specified bucket.
    """
    bucket = client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=directory_name)
    return [blob.name for blob in blobs]


def download_file_from_bucket(
    bucket: Bucket, file_path: str, download_directory_path
) -> str:
    """
    Downloads a file from a Google Cloud Storage bucket to a local directory.

    Args:
        bucket (Bucket): The Google Cloud Storage bucket object.
        file_path (str): The path to the file within the bucket.
        download_directory_path (str): The local directory path where the file will be downloaded.

    Returns:
        str: The local file path where the file has been downloaded.

    Raises:
        google.cloud.exceptions.NotFound: If the file does not exist in the bucket.
        google.cloud.exceptions.GoogleCloudError: If there is an error during the download process.

    Example:
        local_path = download_file_from_bucket(bucket, 'path/to/file.txt', '/local/download/directory')
    """
    try:
        blob = bucket.blob(file_path)
        local_file_name = os.path.basename(file_path)
        local_filepath = os.path.join(download_directory_path, local_file_name)
        try:
            blob.download_to_filename(local_filepath)
            print(f"Downloaded '{file_path}' to '{local_file_name}'")
            return local_filepath
        except GoogleCloudError:
            print("Error during download process")
    except NotFound:
        print("Error: File does not exist in the bucket")


def create_cloud_sql_database_connection() -> PostgresEngine:
    """
    Establishes a connection to a Cloud SQL PostgreSQL database instance.

    Returns:
        PostgresEngine: An instance of PostgresEngine connected to the specified Cloud SQL database.

    Raises:
        SomeException: Description of the exception that might be raised (if applicable).

    Example:
        connection = create_cloud_sql_database_connection()
    """

    engine = PostgresEngine.from_instance(
        project_id=PROJECT_ID,
        instance=INSTANCE,
        region=REGION,
        database=DATABASE,
        user=DB_USER,
        password=DB_PASSWORD,
    )

    return engine


async def create_table_if_not_exists(table_name: str, engine: PostgresEngine) -> None:
    """
    Creates a table in the vector store if it does not already exist.

    This function attempts to initialize a vector store table with the specified
    table name and a vector size of 768, which is suitable for the VertexAI model
    (textembedding-gecko@latest). If the table already exists, it catches the
    ProgrammingError and prints a message indicating that the table is already created.

    Args:
        table_name (str): The name of the table to be created.
        engine (PostgresEngine): The database engine to create the table in.

    Raises:
        ProgrammingError: If the table already exists.
    """
    try:
        await engine.init_vectorstore_table(
            table_name=table_name,
            vector_size=768,
        )
    except ProgrammingError:
        print("Table already created")


def get_embeddings() -> VertexAIEmbeddings:
    """
    Retrieves the VertexAIEmbeddings instance for the specified model.

    Returns:
        VertexAIEmbeddings: An instance of VertexAIEmbeddings configured with the specified model and project.

    Example:
        embeddings = get_embeddings()
    """
    embeddings = VertexAIEmbeddings(
        model_name="textembedding-gecko@latest", project=PROJECT_ID
    )
    return embeddings


def get_vector_store(
    engine: PostgresEngine, table_name: str, embedding: VertexAIEmbeddings
) -> PostgresVectorStore:
    """
    Retrieves the vector store from the specified database engine.

    Args:
        engine (PostgresEngine): The database engine to retrieve the vector store from.
        table_name (str): The name of the table to retrieve the vector store from.
        embedding (VertexAIEmbeddings): The VertexAIEmbeddings instance to use for the vector store.

    Returns:
        VectorStore: The vector store object.

    Example:
        vector_store = get_vector_store(engine, 'my_table', embedding)
    """
    vector_store = PostgresVectorStore.create_sync(  # Use .create() to initialize an async vector store
        engine=engine,
        table_name=table_name,
        embedding_service=embedding,
    )
    return vector_store
