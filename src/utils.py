# ===================================================================================
# Project: SpectraRAG
# File: src/utils.py
# Description: This file contains various utility functions used in the project.
# Author: LALAN KUMAR
# Created: [22-07-2025]
# Updated: [24-07-2025]
# LAST MODIFIED BY: LALAN KUMAR [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

import os
import sys
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
import hashlib
import uuid
from langchain.schema import Document
from typing import Any, Union, Literal

# Dynamically add the project root directory to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from config import settings

def get_embeddings(provider: Optional[str] = None) -> Embeddings:
    """Initialize and return embeddings instance based on provider.
    
    Args:
        provider: The embeddings provider to use. If None, uses the default from settings.
        
    Returns:
        A LangChain embeddings instance.
        
    Raises:
        ValueError: If the requested provider is not supported.
    """
    # Use default provider if none specified
    if provider is None:
        provider = settings.EMBEDDING_PROVIDER
    
    # Return the appropriate embeddings based on provider
    if provider == settings.EMBEDDING_PROVIDER_GEMINI:
        if not settings.GEMINI_API_KEY:
            raise ValueError("Gemini API key not found in environment")
        return GoogleGenerativeAIEmbeddings(model=settings.GEMINI_EMBEDDING_MODEL)
    
    elif provider == settings.EMBEDDING_PROVIDER_OPENAI:
        if not settings.OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found in environment")
        return OpenAIEmbeddings(model=settings.OPENAI_EMBEDDING_MODEL)
    
    else:
        raise ValueError(f"Unsupported embeddings provider: {provider}")
    
    
def get_llm(provider: Optional[str] = None) -> BaseChatModel:
    """Initialize and return LLM instance based on provider.
    
    Args:
        provider: The LLM provider to use. If None, uses the default from settings.
        
    Returns:
        A LangChain chat model instance.
        
    Raises:
        ValueError: If the requested provider is not supported.
    """
    # Use default provider if none specified
    if provider is None:
        provider = settings.LLM_PROVIDER
    
    # Return the appropriate LLM based on provider
    if provider == settings.LLM_PROVIDER_GEMINI:
        if not settings.GEMINI_API_KEY:
            raise ValueError("Gemini API key not found in environment")
        return ChatGoogleGenerativeAI(model=settings.GEMINI_LLM_MODEL)
    
    elif provider == settings.LLM_PROVIDER_OPENAI:
        if not settings.OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found in environment")
        return ChatOpenAI(model=settings.OPENAI_LLM_MODEL)
    
    elif provider == settings.LLM_PROVIDER_ANTHROPIC:
        if not settings.ANTHROPIC_API_KEY:
            raise ValueError("Anthropic API key not found in environment")
        return ChatAnthropic(model=settings.ANTHROPIC_LLM_MODEL)
    
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
    
    
def _generate_uuid(page_content: str) -> str:
    """Generate a UUID for a document based on page content."""
    md5_hash = hashlib.md5(page_content.encode()).hexdigest()
    return str(uuid.UUID(md5_hash))

def reduce_docs(
    existing: Optional[list[Document]],
    new: Union[
        list[Document],
        list[dict[str, Any]],
        list[str],
        str,
        Literal["delete"],
    ],
) -> list[Document]:
    """Reduce and process documents based on the input type.

    This function handles various input types and converts them into a sequence of Document objects.
    It can delete existing documents, create new ones from strings or dictionaries, or return the existing documents.
    It also combines existing documents with the new one based on the document ID.

    Args:
        existing (Optional[Sequence[Document]]): The existing docs in the state, if any.
        new (Union[Sequence[Document], Sequence[dict[str, Any]], Sequence[str], str, Literal["delete"]]):
            The new input to process. Can be a sequence of Documents, dictionaries, strings, a single string,
            or the literal "delete".
    """
    if new == "delete":
        return []

    existing_list = list(existing) if existing else []
    if isinstance(new, str):
        return existing_list + [
            Document(page_content=new, metadata={"uuid": _generate_uuid(new)})
        ]

    new_list = []
    if isinstance(new, list):
        existing_ids = set(doc.metadata.get("uuid") for doc in existing_list)
        for item in new:
            if isinstance(item, str):
                item_id = _generate_uuid(item)
                new_list.append(Document(page_content=item, metadata={"uuid": item_id}))
                existing_ids.add(item_id)

            elif isinstance(item, dict):
                metadata = item.get("metadata", {})
                item_id = metadata.get("uuid") or _generate_uuid(
                    item.get("page_content", "")
                )

                if item_id not in existing_ids:
                    new_list.append(
                        Document(**{**item, "metadata": {**metadata, "uuid": item_id}})
                    )
                    existing_ids.add(item_id)

            elif isinstance(item, Document):
                item_id = item.metadata.get("uuid", "")
                if not item_id:
                    item_id = _generate_uuid(item.page_content)
                    new_item = item.model_copy(deep=True)
                    new_item.metadata["uuid"] = item_id
                else:
                    new_item = item

                if item_id not in existing_ids:
                    new_list.append(new_item)
                    existing_ids.add(item_id)

    return existing_list + new_list

def format_docs(docs: Optional[list[Document]]) -> str:
    """Format a list of documents as XML.

    Args:
        docs (Optional[list[Document]]): A list of Document objects to format, or None.

    Returns:
        str: A string containing the formatted documents in XML format.
    """
    if not docs:
        return "<documents></documents>"
    
    formatted = "\n".join(_format_doc(doc) for doc in docs)
    return f"""<documents>
{formatted}
</documents>"""

def _format_doc(doc: Document) -> str:
    """Format a single document as XML with special handling for code blocks."""
    metadata = doc.metadata or {}
    meta = "".join(f" {k}={v!r}" for k, v in metadata.items())
    if meta:
        meta = f" {meta}"

    # Process the page content to preserve code blocks
    page_content = doc.page_content
    
    # Add special xml tags to mark code blocks for the LLM
    return f"<document{meta}>\n<content>\n{page_content}\n</content>\n</document>"


# ===================================================================================
# ui_utils.py
# ===================================================================================
import streamlit as st
import os
import shutil
import time

def apply_custom_styling():
    """Apply custom styling to the Streamlit app"""
    st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    /* Chat Input Styling */
    .stChatInput input {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 1px solid #3A3A3A !important;
    }
    
    /* User Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #1E1E1E !important;
        border: 1px solid #3A3A3A !important;
        color: #E0E0E0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Assistant Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #2A2A2A !important;
        border: 1px solid #404040 !important;
        color: #F0F0F0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Avatar Styling */
    .stChatMessage .avatar {
        background-color: #00FFAA !important;
        color: #000000 !important;
    }
    
    /* Text Color Fix */
    .stChatMessage p, .stChatMessage div {
        color: #FFFFFF !important;
    }
    
    .stFileUploader {
        background-color: #1E1E1E;
        border: 1px solid #3A3A3A;
        border-radius: 5px;
        padding: 15px;
    }
    
    h1, h2, h3 {
        color: #00FFAA !important;
    }
    
    .stButton button {
        background-color: #1E1E1E !important;
        color: #00FFAA !important;
        border: 1px solid #3A3A3A !important;
        border-radius: 5px;
        padding: 5px 10px;
        font-weight: bold;
        font-size: 16px;
    }
    
    .stButton button:hover {
        background-color: #2A2A2A !important;
        border: 1px solid #00FFAA !important;
    }
    
    .document-list {
        margin-top: 10px;
    }
    
    .document-item {
        padding: 8px;
        margin: 4px 0;
        background-color: #1E1E1E;
        border-radius: 4px;
        border-left: 3px solid #00FFAA;
    }
    
    .file-badge {
        display: inline-block;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
        margin-left: 6px;
    }
    
    .file-badge-pdf {
        background-color: #FF5252;
        color: white;
    }
    
    .file-badge-docx {
        background-color: #2196F3;
        color: white;
    }
    
    .file-badge-xlsx {
        background-color: #4CAF50;
        color: white;
    }
    
    .file-badge-txt {
        background-color: #9E9E9E;
        color: white;
    }
    
    .document-expander {
        border: 1px solid #3A3A3A;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

def cleanup_session_files():
    """Clean up session files and vector DB when session ends"""
    session_id = st.session_state.get("session_id", None)
    if not session_id:
        return
    
    # Clean up uploaded files
    if "last_uploaded_file" in st.session_state and st.session_state.last_uploaded_file:
        try:
            if os.path.exists(st.session_state.last_uploaded_file):
                os.remove(st.session_state.last_uploaded_file)
                print(f"Cleaned up uploaded file: {st.session_state.last_uploaded_file}")
        except Exception as e:
            print(f"Error cleaning up uploaded file: {e}")
    
    # Clean up temp_uploads directory
    temp_uploads_dir = "temp_uploads"
    if os.path.exists(temp_uploads_dir):
        try:
            shutil.rmtree(temp_uploads_dir)
            print(f"Cleaned up temp uploads directory: {temp_uploads_dir}")
        except Exception as e:
            print(f"Error cleaning up temp uploads directory: {e}")
    
    # Clean up vector DB
    vector_db_path = f"DATA/vector_db_{session_id}"
    if os.path.exists(vector_db_path):
        try:
            shutil.rmtree(vector_db_path)
            print(f"Cleaned up vector DB: {vector_db_path}")
        except Exception as e:
            print(f"Error cleaning up vector DB: {e}")
    
    # Clean up retriever cache
    retriever_path = f"DATA/retriever_cache_{session_id}"
    if os.path.exists(retriever_path):
        try:
            shutil.rmtree(retriever_path)
            print(f"Cleaned up retriever cache: {retriever_path}")
        except Exception as e:
            print(f"Error cleaning up retriever cache: {e}")
    
    # Clean up message bus
    from src.mcp.message_protocol import message_buses
    if session_id in message_buses:
        del message_buses[session_id]
        print(f"Cleaned up message bus for session: {session_id}")
    
    # Reset session state
    session_keys_to_reset = [
        "has_documents", "uploaded_files", "uploaded_file_info", 
        "last_uploaded_file", "embedding_status", "embedding_error",
        "vector_store", "document_contents", "vector_db_exists"
    ]
    for key in session_keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]

def register_session_cleanup():
    """Register cleanup function to run when session ends"""
    import atexit
    atexit.register(cleanup_session_files)

def initialize_session_state():
    """Initialize the session state variables"""
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    
    if "document_contents" not in st.session_state:
        st.session_state.document_contents = {}
    
    if "message_log" not in st.session_state:
        st.session_state.message_log = [
            {"role": "ai", "content": "Hi! I'm SPECTR, your AI assistant with multi-document intelligence. What can I do for you today? You can chat with me or upload documents (PDF, DOCX, CSV, PPTX, TXT, and more) for analysis."}
        ]
    
    if "processing" not in st.session_state:
        st.session_state.processing = False
    
    if "has_documents" not in st.session_state:
        st.session_state.has_documents = False
    
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
        
    if "uploaded_file_info" not in st.session_state:
        st.session_state.uploaded_file_info = {}
    
    if "last_uploaded_file" not in st.session_state:
        st.session_state.last_uploaded_file = None
    
    if "show_uploader" not in st.session_state:
        st.session_state.show_uploader = False
    
    if "session_id" not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())
    
    # Register cleanup function
    register_session_cleanup()

def display_file_badge(file_type):
    """Generate HTML for a file type badge"""
    badge_class = "file-badge"
    if file_type.lower() in ["pdf"]:
        badge_class += " file-badge-pdf"
    elif file_type.lower() in ["docx", "doc"]:
        badge_class += " file-badge-docx"
    elif file_type.lower() in ["xlsx", "xls", "csv"]:
        badge_class += " file-badge-xlsx"
    else:
        badge_class += " file-badge-txt"
        
    return f'<span class="{badge_class}">{file_type}</span>'