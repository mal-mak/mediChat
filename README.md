# MediChat: RAG-based Medical Question Answering System

A medical chatbot using Retrieval-Augmented Generation (RAG) to provide accurate, source-based medical information that answers the user's questions.

## 🌟 Features

- **RAG Architecture**: Combines retrieval-based and generative approaches for accurate answers
- **Source Transparency**: Provides reference sources with relevance scores for each answer
- **Multiple Languages**: Can answer in English or in French. The input can be any language
- **Evaluation System**: Includes comprehensive evaluation metrics for answer quality
- **Cloud Infrastructure**: Utilizes Google Cloud Platform services
- **Vector Storage**: PostgreSQL-based vector storage for efficient similarity search

## 🏗️ Architecture

```plaintext
                                     ┌─────────────────┐
                                     │  Google Cloud   │
                                     │    Storage      │
                                     └────────┬────────┘
                                              │
                                              ▼
┌─────────────┐    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   FastAPI   │    │  LangChain  │    │  PostgreSQL  │    │   Vertex    │
│    API      │◄─► │     RAG     │◄─► │    Vector    │◄─► │     AI      │
└─────────────┘    └─────────────┘    └──────────────┘    └─────────────┘
        ▲
        │
        ▼
┌─────────────┐
│  Streamlit  │
│ Interface   │
└─────────────┘
```

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **Backend**: FastAPI
- **Database**: PostgreSQL with pgvector
- **Cloud Services**:
  - Google Cloud Storage
  - Google Cloud SQL
  - Vertex AI
- **Key Libraries**:
  - LangChain
  - SentenceTransformers
  - Rich (for CLI output)

## 📁 Project Structure

```
mediChat/
├── eval/                             # detailed eval results
│   └── results/
│       ├── detailed_evaluation_[timestamp].txt
│       └── evaluation_results_[timestamp].json
├── src/
│   └── medichat/
│       ├── api.py                    # FastAPI backend
│       ├── app.py                    # Streamlit frontend
│       ├── eval.py                   # Evaluation system
│       ├── ingest.py                 # Data ingestion
│       └── retrieve.py               # Document retrieval
│       └── gcs_to_cloudsql.ipynb     # notebook for data transfer
└── pyproject.toml                    # Poetry dependencies
└── .env                              # api key and db password
```

## 🚀 Getting Started

### **1️⃣ Environment Setup with Poetry**

```bash
# Clone the repository
git clone https://github.com/mal-mak/mediChat.git
cd mediChat

# Install dependencies using Poetry
poetry install
```

### **2️⃣ Configure Environment Variables**

Create a **`.env`** file and add the following:

```plaintext
GOOGLE_API_KEY="your-api-key"
DB_PASSWORD="your-db-password"
```

### **3️⃣ Running the Application**

```bash
# Set the correct host in app.py to 
HOST = "http://0.0.0.0:8181/"

# Start the API
uvicorn src.medichat.api:app --host 0.0.0.0 --port 8181

# Start the Streamlit interface (in another terminal)
streamlit run src/medichat/app.py
```

## 📊 Evaluation System

The project includes a comprehensive evaluation system (`eval.py`) that measures:

- **Answer Similarity**: Semantic similarity between bot answers and source content
- **Response Time**: Time taken to generate responses
- **Detailed Comparisons**: Saved in both JSON and text formats

Run evaluation:

```bash
poetry run python src/medichat/eval.py
```

## 🎛️ Customization

- **Temperature**: Controls response creativity (0.0-2.0)
- **Similarity Threshold**: Sets minimum relevance score (0.0-1.0)
- **Max Sources**: Maximum number of reference sources (1-20)
- **Language**: English or French responses

## 📝 API Endpoints

- `POST /get_sources`: Retrieves relevant medical documents
- `POST /answer`: Generates answers based on retrieved documents
- `POST /get_files_names`: Lists available reference files

## 🔍 Data Sources

The system uses the [MedQuAD dataset](https://www.kaggle.com/datasets/jpmiller/layoutlm/data), containing:

- Medical questions and answers
- Source attributions
- Focus areas

## 📈 Performance Metrics

Typical performance metrics:

- **Answer Similarity**: ~0.75-0.85
- **Response Time**: 2-6 seconds
- **Source Relevance**: >0.75 threshold

## 📄 License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.