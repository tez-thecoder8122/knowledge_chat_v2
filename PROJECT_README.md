# Knowledge Chat System

A multi-modal Retrieval Augmented Generation (RAG) system that enables intelligent document Q&A with support for extracting, indexing, and retrieving **text, images, and tables** from PDF documents. Built with FastAPI, PostgreSQL, FAISS, and Streamlit.

---

## **Overview**

The Knowledge Chat System allows users to:
- Upload PDF documents and automatically extract text, images, and tables
- Ask natural language questions about the documents
- Receive AI-powered answers with relevant text, images, and tables as evidence
- Visualize results through an intuitive web interface

### **Key Features**
- 🔍 **Semantic Search:** Uses OpenAI embeddings + FAISS for fast vector similarity search
- 🖼️ **Multi-modal Results:** Returns answers with images and tables extracted from PDFs
- 🔐 **Authentication:** JWT-based user authentication and document access control
- 📊 **Database:** PostgreSQL for metadata, FAISS indexes for embeddings
- 🌐 **Web UI:** Streamlit frontend for easy document upload and querying
- 🏗️ **Modular Architecture:** Separate services for document processing, querying, and vision

---

## **Project Structure**

```
knowledge-chat-system/
│
├── app/                          # FastAPI Backend
│   ├── routes/
│   │   ├── auth.py              # Authentication endpoints
│   │   ├── documents.py         # Document upload endpoints
│   │   └── query.py             # Query endpoints with media support
│   ├── services/
│   │   ├── document_service.py  # Document chunking & indexing
│   │   ├── query_service.py     # Query & retrieval logic (with media)
│   │   ├── vision_service.py    # Image & table extraction
│   │   ├── embedding_service.py # OpenAI embeddings
│   │   └── auth_service.py      # JWT authentication
│   ├── models/
│   │   ├── database.py          # SQLAlchemy models
│   │   └── schemas.py           # Pydantic schemas
│   ├── db/
│   │   └── session.py           # Database session management
│   ├── config.py                # Environment configuration
│   ├── main.py                  # FastAPI app entry point
│   └── utils/
│       └── logger.py            # Logging setup
│
├── frontend/                     # Streamlit Frontend
│   ├── app.py                   # Main Streamlit application
│   ├── requirements.txt         # Frontend dependencies
│   └── README.md               # Frontend-specific docs
│
├── docker-compose.yml           # Multi-container orchestration
├── Dockerfile                   # Backend container configuration
├── requirements.txt             # Backend dependencies
├── .env.example                 # Environment variables template
├── alembic.ini                  # Database migration config
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

---

## **Prerequisites**

- **Docker & Docker Compose** (for containerized setup)
- **Python 3.11+** (for local development)
- **OpenAI API Key** (for embeddings and LLM)
- **PostgreSQL 13+** (optional, if running locally)

---

## **Quick Start**

### **Option 1: Docker Compose (Recommended)**

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd knowledge-chat-system
   ```

2. **Create `.env` file from template:**
   ```bash
   cp .env.example .env
   ```
   Update `.env` with your **OpenAI API Key** and other settings:
   ```
   OPENAI_API_KEY=sk-your-key-here
   DATABASE_URL=postgresql://user:password@db:5432/knowledge_chat
   SECRET_KEY=your-secret-key
   ```

3. **Start all services (Backend + Database):**
   ```bash
   docker-compose up --build
   ```
   - Backend API runs on `http://localhost:8000`
   - PostgreSQL runs on `localhost:5432`
   - Check logs: `docker-compose logs -f knowledge_chat_api`

4. **In a new terminal, start Streamlit frontend:**
   ```bash
   cd frontend
   pip install -r requirements.txt
   streamlit run app.py
   ```
   - Streamlit opens at `http://localhost:8501`

5. **Access the application:**
   - **Frontend:** http://localhost:8501
   - **Backend API Docs:** http://localhost:8000/docs

---

### **Option 2: Local Development (Without Docker)**

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd knowledge-chat-system
   ```

2. **Create Python virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install backend dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API Key and database URL
   ```

5. **Ensure PostgreSQL is running** on your machine (or update `DATABASE_URL` in `.env`)

6. **Start the backend API:**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

7. **In another terminal, install and run Streamlit:**
   ```bash
   cd frontend
   pip install -r requirements.txt
   streamlit run app.py
   ```

---

## **Usage**

### **Via Streamlit Frontend**

1. **Login:**
   - Default credentials: `username: alice`, `password: password123` (or your configured user)
   - Sign in using the sidebar

2. **Upload Documents:**
   - Go to **"Upload Documents"** tab
   - Select a PDF file (max 200MB per file)
   - Click **"Upload Document"**
   - Backend automatically extracts text, chunks, embeds, and extracts images/tables

3. **Ask Questions:**
   - Go to **"Ask Questions"** tab
   - Enter a natural language question
   - Adjust **"Number of context chunks"** (1-10, default 3)
   - Click **"Get Answer"**
   - View:
     - **Answer** generated by LLM
     - **Text Sources** with relevance scores
     - **Related Media** (images and tables)

---

## **Testing with Postman**

### **1. Authenticate:**
- POST `http://localhost:8000/auth/login`
- Body (raw/JSON):
  ```json
  {
    "username": "alice",
    "password": "password123"
  }
  ```
- Copy `access_token` from response

### **2. Upload Document:**
- POST `http://localhost:8000/documents/upload`
- Headers: `Authorization: Bearer <your-token>`
- Body: form-data, key `file`, select PDF

### **3. Query with Media:**
- POST `http://localhost:8000/api/query/ask`
- Headers: `Authorization: Bearer <your-token>`, `Content-Type: application/json`
- Body:
  ```json
  {
    "question": "Show me the specifications table from the manual.",
    "top_k": 5,
    "include_media": true
  }
  ```
- Response includes `media_items` with images (base64) and tables (HTML/CSV)

### **4. View Media Details:**
- GET `http://localhost:8000/api/query/media/{media_id}`
- GET `http://localhost:8000/api/query/media/{media_id}/data`

---

## **Environment Variables**

Create a `.env` file in the root directory:

```
# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-key

# Database
DATABASE_URL=postgresql://username:password@localhost:5432/knowledge_chat
SQLALCHEMY_TRACK_MODIFICATIONS=False

# JWT Authentication
SECRET_KEY=your-secret-key-for-jwt
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Server
DEBUG=True
ENVIRONMENT=development

# Embedding & Search
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4
MAX_TOKENS=1200
TEMPERATURE=0.3
```

---

## **Database Schema**

Key tables:
- **users:** User accounts and authentication
- **documents:** Uploaded PDF metadata
- **document_chunks:** Text chunks with embeddings
- **document_media:** Extracted images and tables
- **queries:** Query history and analytics

---

## **API Endpoints**

### **Authentication**
- `POST /auth/login` - User login
- `POST /auth/register` - User registration

### **Documents**
- `POST /documents/upload` - Upload and process PDF
- `GET /documents/list` - List user's documents

### **Query**
- `POST /api/query/ask` - Ask a question with media support
- `GET /api/query/media/{media_id}` - Get media details
- `GET /api/query/media/{media_id}/data` - Download media

Full API docs available at `http://localhost:8000/docs` (Swagger UI)

---

## **Troubleshooting**

### **"Upload failed: Unknown error" in Streamlit**
- Check backend logs: `docker-compose logs knowledge_chat_api`
- Ensure PDF is valid and under 200MB
- Verify OpenAI API key is set

### **"Unable to get page count. Is poppler installed?" Error**
- Poppler is installed in the Docker image by default
- For local development on Ubuntu: `sudo apt-get install poppler-utils`
- On macOS: `brew install poppler`

### **No media items in query response**
- Ensure `include_media: true` is set in the query
- Check that media was extracted during document upload (check logs)
- Try a more general query ("Show me images from page 1")

### **Database connection errors**
- Verify PostgreSQL is running
- Check `DATABASE_URL` in `.env`
- Run migrations if needed: `alembic upgrade head`

### **OpenAI API errors**
- Verify API key is valid and has sufficient quota
- Check rate limits: https://platform.openai.com/account/rate-limits

---

## **Performance Optimization**

- **FAISS Index:** Indexes are cached on disk for fast retrieval
- **Chunk Size:** Tune `CHUNK_SIZE` and `CHUNK_OVERLAP` in config for quality vs. speed
- **Embedding Cache:** Reuse embeddings for identical chunks
- **Database Indexing:** Ensure PostgreSQL indexes are created (handled by migrations)

---

## **Development & Contributing**

### **Code Structure:**
- **Services:** Business logic separated from routes
- **Models:** Database and schema definitions
- **Routes:** FastAPI endpoint handlers

### **Adding New Features:**
1. Create new service in `app/services/`
2. Add routes in `app/routes/`
3. Update schemas in `app/models/schemas.py`
4. Test with Postman or Streamlit

### **Running Tests:**
```bash
pytest tests/ -v
```

---

## **Deployment**

### **Docker Hub:**
```bash
docker build -t knowledge-chat-system:latest .
docker tag knowledge-chat-system:latest your-registry/knowledge-chat-system:latest
docker push your-registry/knowledge-chat-system:latest
```

### **Cloud Deployment (AWS/GCP/Azure):**
- Use `docker-compose.yml` as reference
- Scale PostgreSQL with managed services (RDS, Cloud SQL, etc.)
- Use serverless for Streamlit or deploy on ECS/Kubernetes

---

**Happy querying! 🚀**
