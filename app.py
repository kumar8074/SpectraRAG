import os
import shutil
import asyncio
import streamlit as st
from src.pipeline.mcp_agent_spectr import MCPSpectraRagController
from src.utils import apply_custom_styling, initialize_session_state, display_file_badge, cleanup_session_files

TEMP_UPLOADS = "temp_uploads"

st.set_page_config(page_title="SpectraRAG", layout="wide")
apply_custom_styling()
initialize_session_state()

st.title("SpectraRAG: Multi-Agent RAG with MCP")
st.caption("Your AI Assistant with Multi-Document Intelligence")

# --- Chat Container ---
chat_container = st.container()

# --- Upload and Chat Input Row ---
input_container = st.container()
col1, col2 = input_container.columns([10, 2])

with col2:
    col2_1, col2_2 = st.columns(2)
    
    with col2_1:
        upload_toggle = st.button("📄 +", key="upload_toggle_btn")
        if upload_toggle:
            st.session_state.show_uploader = not st.session_state.get("show_uploader", False)
    
    with col2_2:
        if st.session_state.get("has_documents", False):
            if st.button("🗑️", key="cleanup_btn", help="Clear session and cleanup files"):
                cleanup_session_files()
                st.success("Session cleaned up successfully!")
                st.rerun()

with col1:
    embedding_status = st.session_state.get("embedding_status", None)
    embedding_error = st.session_state.get("embedding_error", None)
    has_documents = st.session_state.get("has_documents", False)
    # Chat input is always visible after upload, but send is only enabled when ready
    chat_disabled = False
    send_disabled = False
    send_help = None
    if has_documents:
        if embedding_status == "processing":
            chat_disabled = False
            send_disabled = True
            send_help = "Document is processing. Please wait for vectorDB to be ready."
        elif embedding_status == "error":
            chat_disabled = False
            send_disabled = True
            send_help = f"VectorDB creation failed: {embedding_error}"
        elif embedding_status == "ready":
            chat_disabled = False
            send_disabled = False
        else:
            chat_disabled = False
            send_disabled = True
            send_help = "Upload document to enable chat."
    else:
        # No documents uploaded: chat and send enabled as normal
        chat_disabled = False
        send_disabled = False
    # Custom chat input with send button
    with st.form("chat_form", clear_on_submit=True):
        user_query = st.text_input("Ask SpectraRAG...", key="chat_input", disabled=chat_disabled)
        send_btn = st.form_submit_button("Send", disabled=send_disabled, help=send_help)
    if send_disabled and send_help:
        st.caption(f"**{send_help}**")
    if not user_query:
        user_query = None
    elif not send_btn:
        user_query = None


# --- Show file uploader when toggled ---
supported_file_types = ["pdf", "docx", "pptx", "txt", "csv", "md", "log"]
if st.session_state.get("show_uploader", False):
    uploaded_file = st.file_uploader(
        "Upload Document",
        type=supported_file_types,
        key="file_uploader",
        label_visibility="visible"
    )
    if uploaded_file and (st.session_state.get("last_uploaded_file", None) != uploaded_file.name):
        with st.spinner("Processing document and creating vectorDB..."):
            try:
                os.makedirs(TEMP_UPLOADS, exist_ok=True)
                file_save_path = os.path.join(TEMP_UPLOADS, uploaded_file.name)
                with open(file_save_path, "wb") as f:
                    shutil.copyfileobj(uploaded_file, f)
                st.session_state["last_uploaded_file"] = file_save_path
                st.session_state["has_documents"] = True
                st.session_state["uploaded_files"] = [uploaded_file.name]
                st.session_state["uploaded_file_info"] = {
                    uploaded_file.name: {
                        "file_type": uploaded_file.type,
                        "file_size_kb": int(len(uploaded_file.getvalue()) / 1024)
                    }
                }
                st.session_state["embedding_status"] = "processing"
                st.session_state["embedding_error"] = None
                st.session_state.show_uploader = False

                # Immediately trigger embedding using MCP controller
                async def async_embed():
                    try:
                        controller = MCPSpectraRagController()
                        # Only embedding, no query
                        result = await controller.run(st.session_state["last_uploaded_file"], "__EMBED_ONLY__")
                        if result.get("error"):
                            st.session_state["embedding_status"] = "error"
                            st.session_state["embedding_error"] = result["error"]
                            st.session_state["has_documents"] = False
                        elif result.get("vector_db_ready"):
                            st.session_state["embedding_status"] = "ready"
                            st.session_state["embedding_error"] = None
                            st.session_state["message_log"].append({
                                "role": "ai",
                                "content": f"✅ VectorDB created and ready for question answering."
                            })
                        else:
                            st.session_state["embedding_status"] = "processing"
                    except Exception as e:
                        st.session_state["embedding_status"] = "error"
                        st.session_state["embedding_error"] = str(e)
                        st.session_state["has_documents"] = False
                    finally:
                        st.rerun()
                asyncio.run(async_embed())
                st.rerun()
            except Exception as e:
                st.session_state["embedding_status"] = "error"
                st.session_state["embedding_error"] = str(e)
                st.session_state["has_documents"] = False
                st.error(f"Failed to process document '{uploaded_file.name}': {str(e)}")

# --- Uploaded Documents Expander ---
if st.session_state.get("uploaded_files"):
    with st.expander("📚 Uploaded Documents"):
        for file_name in st.session_state["uploaded_files"]:
            file_info = st.session_state["uploaded_file_info"].get(file_name, {})
            file_type = file_info.get("file_type", "Unknown")
            file_size = file_info.get("file_size_kb", 0)
            st.write(f"- {file_name} ({file_type}, {file_size} KB) {display_file_badge(file_name.split('.')[-1])}")

# --- Display Chat History ---
with chat_container:
    for message in st.session_state["message_log"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if st.session_state.get("processing"):
        with st.chat_message("ai"):
            st.write("Processing...")

# --- Handle User Query Submission ---
if user_query:
    st.session_state["message_log"].append({"role": "user", "content": user_query})
    st.session_state["processing"] = True
    st.rerun()

# --- Process Query with RAG ---
if st.session_state.get("processing"):
    with st.spinner("Running RAG pipeline..."):
        async def async_run():
            try:
                controller = MCPSpectraRagController()
                
                # Determine if we have documents and should use RAG pipeline
                has_documents = st.session_state.get("has_documents", False)
                embedding_ready = st.session_state.get("embedding_status") == "ready"
                file_path = st.session_state.get("last_uploaded_file")
                query = st.session_state["message_log"][-1]["content"]
                
                # Debug logging
                print(f"DEBUG: has_documents={has_documents}, embedding_ready={embedding_ready}, file_path={file_path}")
                print(f"DEBUG: query={query}")
                
                # Only pass file_path if we have documents and embedding is ready
                if has_documents and embedding_ready and file_path:
                    print(f"DEBUG: Using RAG pipeline with file_path={file_path}")
                    result = await controller.run(file_path, query)
                else:
                    print(f"DEBUG: Using GeneralAgent with file_path=None")
                    # Route to general agent by passing None for file_path
                    result = await controller.run(None, query)
                if "error" in result:
                    ai_response = f"❌ Error: {result['error']}"
                else:
                    ai_response = f"{result['answer']}"
                    if result.get("source_context"):
                        ai_response += f"\n\n<details><summary>Show Source Context</summary>\n\n" + f"```\n{result['source_context']}\n```" + "\n</details>"
                    if result.get("trace_id"):
                        ai_response += f"\n<sub>Trace ID: {result['trace_id']}</sub>"
                st.session_state["message_log"].append({"role": "ai", "content": ai_response})
            except Exception as e:
                st.session_state["message_log"].append({"role": "ai", "content": f"❌ Exception: {e}"})
            finally:
                st.session_state["processing"] = False
                st.rerun()
        asyncio.run(async_run())
