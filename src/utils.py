# ===================================================================================
# Project: SpectraRAG
# File: src/utils.py
# Description: This file contains various utility functions used in the project.
# Author: LALAN KUMAR
# Created: [22-07-2025]
# Updated: [22-07-2025]
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