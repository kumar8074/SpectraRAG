# ===================================================================================
# Project: SpectraRAG
# File: src/components/prompts.py
# Description: This file contains the system prompts used by the graphs.
# Author: LALAN KUMAR
# Created: [22-07-2025]
# Updated: [22-07-2025]
# LAST MODIFIED BY: LALAN KUMAR [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

GENERATE_QUERIES_SYSTEM_PROMPT = """\
Generate 3 search queries to search for to answer the user's question. \
These search queries should be diverse in nature - do not generate \
repetitive ones."""

# System prompt for general queries
GENERAL_SYSTEM_PROMPT = "You are a helpful Assistant, Answer the user's question using your knowledge."