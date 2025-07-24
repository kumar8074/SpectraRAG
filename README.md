# SpectraRAG

**A Multi-Agent, Multi-Document Conversational Retrieval-Augmented Generation (RAG) System with Modular Agent Coordination**

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Directory Structure](#directory-structure)
- [Implementation Details](#implementation-details)
  - [Agent Design](#agent-design)
  - [Session and State Management](#session-and-state-management)
  - [Workflow: End-to-End Pipeline](#workflow-end-to-end-pipeline)
- [Usage](#usage)
- [Development & Customization](#development--customization)
- [Acknowledgements](#acknowledgements)

---

## Overview

SpectraRAG is a modular, multi-agent document chat system that enables users to upload documents, have them embedded and indexed into a vector database, and then query them conversationally using a multi-turn chat interface. The system is designed for extensibility, reliability, and clear separation of concerns, using a message-passing architecture to coordinate specialized agents for ingestion, retrieval, response generation, and general knowledge queries.

---

## System Architecture

### High-Level Flow

1. **User uploads a document** via the Streamlit UI.
2. **Embedder Agent** is triggered to embed the document and persist the vector database.
3. **UI disables chat input** until embedding is complete.
4. **Retriever Agent** is triggered on user queries (after embedding), retrieving relevant documents from the vectorDB.
5. **Response Agent** generates answers based on the retrieved context and user query.
6. **General Agent** handles queries if no document is uploaded.
7. **Session Management** ensures that vectorDB and uploaded files are cleaned up at session end, and that retriever instances persist per session.

### Component Diagram

```
[User/Streamlit UI]
        │
        ▼
[MCPSpectraRagController]
        │
        ▼
   [MCPCoordinator]
   ┌───────────────┬───────────────┬───────────────┬───────────────┐
   │               │               │               │               │
[Embedder]   [Retriever]   [Response]   [General]
   │               │               │               │
   └───────────────┴───────────────┴───────────────┴───────────────┘
        │
        ▼
   [VectorDB, Session State, File Storage]
```

---

## Key Features

- **Multi-Agent Modular Design:** Separate agents for embedding, retrieval, response, and general knowledge.
- **Session-Aware VectorDB:** Vector database is persisted per session and only created once per upload.
- **Multi-Turn Chat:** Conversation history is maintained for context-aware responses.
- **Asynchronous Message Passing:** Agents communicate via an MCP message bus for robust orchestration.
- **Automatic Cleanup:** Uploaded files and vectorDB are deleted at session end.
- **Fallback to General Agent:** If no document is uploaded, queries are routed to a general-purpose agent.
- **Streamlit UI:** Modern, interactive web interface with upload, chat, and session controls.

---

## Directory Structure

```
SpectraRAG/
│
├── app.py                # Streamlit UI & session logic
├── requirements.txt
├── .gitignore
├── README.md
├── DATA/                 # VectorDB storage (gitignored)
├── temp_uploads/         # Uploaded files (gitignored)
├── logs/                 # Log files (gitignored)
│
├── src/
│   ├── Agents/
│   │   ├── embedder_agent.py
│   │   ├── retriever_agent.py
│   │   ├── response_agent.py
│   │   ├── general_agent.py
│   │   └── __init__.py
│   ├── components/
│   │   ├── states.py     # Pydantic/TypedDict schemas for agent state
│   ├── mcp/
│   │   ├── coordinator.py         # Main agent orchestrator
│   │   ├── mcp_agents.py          # Agent wrappers for MCP
│   │   ├── message_protocol.py    # Message bus and protocol
│   ├── pipeline/
│   │   ├── mcp_agent_spectr.py    # Controller for Streamlit UI
│   │   ├── agent_spectr.py        # (legacy/alt) controller
│   ├── utils.py
│   └── logger.py
│
└── config/
    ├── settings.py
    └── __init__.py
```

---

## Implementation Details

### Agent Design

- **Embedder Agent:** Handles document ingestion, embedding, and vectorDB persistence.
- **Retriever Agent:** Retrieves relevant documents from the vectorDB for a given query.
- **Response Agent:** Generates conversational answers using the retrieved context and user input.
- **General Agent:** Handles queries when no document is uploaded, using general LLM knowledge.

All agents are orchestrated through the `MCPCoordinator` and communicate asynchronously using the MCP message protocol.

### Session and State Management

- **Session-Unique Paths:** Each user/session gets a unique directory for uploads and vectorDB storage.
- **Session State:** Streamlit's `st.session_state` is used to track the current document, vectorDB readiness, chat history, and more.
- **Cleanup:** On session end or when the user clicks "Clear Session," all files and vectorDBs are deleted.

### Workflow: End-to-End Pipeline

1. **Document Upload:**
   - User uploads a file via the UI.
   - The file is saved to a session-unique directory.
   - The embedder agent is triggered via a special query (`__EMBED_ONLY__`).
   - The UI disables chat input until embedding is complete.
   - Once embedding is done, `vector_db_ready` is set in the session state.

2. **Query Processing:**
   - If a document is uploaded and vectorDB is ready:
     - The retriever agent is triggered for every query.
     - Retrieved documents and the query are sent to the response agent.
     - The assistant's answer is appended to the chat history.
   - If no document is uploaded:
     - The general agent is triggered for queries.

3. **Session Cleanup:**
   - When the session ends or the user clicks "Clear Session," all uploaded files and vectorDBs are deleted.

---

## Usage

### 1. Create a Virtual Environment

It's recommended to create a virtual environment to isolate the project dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up API Keys

Create a `.env` file in the project root directory and add your API keys:

```bash
# Copy the example file (if available) or create a new .env file
cp .env.example .env

# Edit the .env file and add your API keys:
# OPENAI_API_KEY=your_openai_api_key_here
# ANTHROPIC_API_KEY=your_anthropic_api_key_here
# Add other required API keys as needed
```

### 4. Run the Application

```bash
streamlit run app.py
```

### 3. Interact

- Upload one or more documents.
- Wait for embedding to complete.
- Ask questions about your documents or general questions.
- Use the "Clear Session" button to remove all files and reset the session.

---

## Development & Customization

- **Add New Agents:** Implement new agent logic in `src/Agents/` and register via MCP.
- **Change VectorDB/Embedding:** Modify the embedder agent and vectorDB handling in `src/Agents/embedder_agent.py`.
- **UI Customization:** Edit `app.py` and `src/utils.py` for new UI features or styling.
- **Logging:** Logs are written to the `logs/` directory.

---

## Acknowledgements

- Built by [Lalan Kumar](https://github.com/kumar8074).
- Uses [LangChain](https://github.com/langchain-ai/langchain), [LangGraph](https://github.com/langchain-ai/langgraph), [Streamlit](https://github.com/streamlit/streamlit), and other open-source technologies.

---

**For any issues or contributions, please open a GitHub issue or PR.**
