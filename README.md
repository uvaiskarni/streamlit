
# AI Butler

## Overview
AI Butler is a tool designed to assist recruiters by providing context-aware responses based on a candidate's experience. It uses Retrieval-Augmented Generation (RAG) with a Large Language Model (LLM) to generate tailored answers from a stored vector database.

## Code Overview

### **AI_butler.py**
- **Purpose**: Main script for the AI Butler application.
- **Features**:
  - Streamlit interface for user interaction.
  - Initializes the vector database and LLM.
  - Handles questions, provides answers using RAG.
  - Maintains logs of activities.
  - Supports GPU cleanup and system shutdown.

### **rag_libs.py**
- **Purpose**: Backend functions for processing documents and generating answers.
- **Features**:
  - Processes documents (PDF, DOCX, TXT).
  - Creates vector embeddings.
  - Interacts with RAG for generating answers.
  - Manages vector database with FAISS for fast similarity searches.

## How to Use
1. **Initialize**: Click "Initialize System" to load the vector database and LLM.
2. **Ask a Question**: Enter your question and submit. Press Ctrl+Enter or use the right-side button.
3. **View Logs**: Check the sidebar for system logs.
4. **Deinitialize**: Use "Deinitialize System" to clean up resources when finished.

## Limitations
- Only one active instance at a time.
- Requires reinitialization after deinitialization.
- Idle timeout resets the system after 5 minutes of inactivity.
- Hosted on ngrok (free version), which may limit resource availability.
