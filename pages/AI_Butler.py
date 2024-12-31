import os
import time
import gc
import json
import torch
import requests
import streamlit as st
from lib.rag_libs_Llama import (
    initialise_llm,
    answer_with_rag,
    load_vector_database,
    generate_save_vector_database,
)

# Constants
CACHE_PATH = "./cache"
DATA_DIR = "./docs"
IDLE_TIMEOUT = 300
LOG_FILE = "session_logs.txt"

# Initialize Session State
def initialize_session_state():
    defaults = {
        "logs": [],
        "last_activity": time.time(),
        "vector_db": None,
        "llm": None,
        "initialized": False,
        "session_history": [],
        "question_input": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

# Logging Utility
def add_log(message):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"[{timestamp}] {message}"
    st.session_state["logs"].append(log_message)
    if len(st.session_state["logs"]) > 50:
        st.session_state["logs"].pop(0)
    save_logs_locally(log_message)

# Logging Utility
def add_log_with_location(question, answer):
    """
    Logs the question, answer, timestamp, and rough location based on IP.
    """
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    try:
        # Get location based on IP
        response = requests.get("https://ipinfo.io", timeout=5)
        location_data = response.json()
        city = location_data.get("city", "Unknown")
        region = location_data.get("region", "Unknown")
        country = location_data.get("country", "Unknown")
        location = f"{city}, {region}, {country}"
    except Exception as e:
        location = "Unknown"

    # Create structured log entry
    log_entry = {
        "timestamp": timestamp,
        "question": question,
        "answer": answer,
        "location": location,
    }

    # Save structured log locally
    save_logs_locally(log_entry)

    # Add to session state logs for display
    st.session_state["logs"].append(f"Answered question: '{question}'")
    if len(st.session_state["logs"]) > 50:  # Keep logs manageable
        st.session_state["logs"].pop(0)

def save_logs_locally(log_entry):
    """
    Save logs to a local file in JSON format.
    """
    try:
        with open(LOG_FILE, "a") as file:
            file.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        st.warning(f"Failed to save logs locally: {e}")

# GPU Memory Cleanup
def reset_gpu_and_clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

# Load or Generate Vector Database
def load_or_generate_vector_db():
    if all(os.path.exists(os.path.join(CACHE_PATH, file)) for file in ["vector_db.index", "docstore.pkl", "index_to_docstore_id.pkl"]):
        add_log("Loading cached vector database...")
        return load_vector_database(CACHE_PATH)
    add_log("Generating new vector database...")
    return generate_save_vector_database(DATA_DIR)

# System Initialization
def initialize_system():
    try:
        st.session_state["vector_db"] = load_or_generate_vector_db()
        st.session_state["llm"] = initialise_llm()
        st.session_state["initialized"] = True
        add_log("System initialized successfully.")
    except Exception as e:
        add_log(f"Initialization failed: {e}")

# System Deinitialization
def deinitialize_system():
    reset_gpu_and_clear_memory()
    for key in ["vector_db", "llm", "logs", "session_history", "question_input"]:
        st.session_state[key] = None if key in ["vector_db", "llm"] else []
    st.session_state["initialized"] = False
    st.session_state["last_activity"] = time.time()
    add_log("System deinitialized successfully.")

# Idle Timeout Check
def check_idle_timeout():
    if time.time() - st.session_state["last_activity"] > IDLE_TIMEOUT:
        add_log("Idle timeout reached. Shutting down...")
        deinitialize_system()
        st.stop()
    st.session_state["last_activity"] = time.time()

# Streamlit Page Configuration
st.set_page_config(page_title="AI Butler", layout="wide")
initialize_session_state()

# Sidebar for Logs
with st.sidebar:
    st.subheader("Logs")
    st.text_area("Logs", value="\n".join(st.session_state["logs"]), height=300)

check_idle_timeout()

# Main Interface
st.title("AI Butler")

# How to Use Section
with st.expander("How to use"):
    st.write("""
    1. Click "Initialize System" to start the service.
    2. Enter your question in the text box and click "Submit" to get a response.
    3. Use the "Deinitialize System" button to release resources.
    """)

# Creating an expander for "about its working"
with st.expander("How RAG with LLM Works in This Code"):
    st.write("""
    **Retrieval-Augmented Generation (RAG) with Large Language Models (LLM)** in this code is designed to provide tailored responses based on the candidate's experience, leveraging document retrieval and AI-driven generation. Here's a detailed breakdown of the process:

    ### 1. **Loading and Processing Data:**
    - **Data Directory**: You provide a directory containing **PDF**, **DOCX**, and **TXT** files with candidate information.
    - The load_process_data() function processes each file type, extracting content from PDFs using PyPDF2, reading Word documents using python-docx, and reading plain text files.
    - The data is structured into **LangchainDocument** objects, where each document contains text and metadata about its source.

    ### 2. **Document Splitting and Tokenization:**
    - After loading the documents, the split_documents() function breaks down large documents into smaller chunks using **RecursiveCharacterTextSplitter**. This is important because LLMs have token limits, and breaking large documents into chunks ensures that the system can process each part efficiently.
    - It uses the tokenizer (AutoTokenizer) for embedding models, and each chunk is processed and prepared for indexing.

    ### 3. **Generating and Storing the Vector Database:**
    - **Embedding**: The documents are converted into vector embeddings using the **HuggingFaceEmbeddings** model. This converts text into dense vector representations, which can be compared based on their similarity.
    - **FAISS Index**: The embeddings are stored in a **FAISS** vector database. This allows for fast, similarity-based search, meaning when a question is asked, the system can retrieve the most relevant pieces of information from the documents.
    - The vector database, along with the document store, is saved into cache files (vector_db.index, docstore.pkl, index_to_docstore_id.pkl).

    ### 4. **Loading the Vector Database:**
    - If a vector database already exists, it can be loaded using the load_vector_database() function. This loads the **FAISS index** and the **docstore** from the cache.
    - The system uses the **HuggingFaceEmbeddings** to map the text into vectors, and the **FAISS** library handles the search for the most relevant documents based on similarity.

    ### 5. **Retrieval Process:**
    - When a recruiter asks a question, the system uses the answer_with_rag() function to retrieve relevant documents.
    - The **similarity search** is performed by the vector database (knowledge_index.similarity_search(query=question, k=num_retrieved_docs)), returning the top documents that are most relevant to the question.
    - These documents are **reranked** using the RERANKER model, ensuring that the best, most relevant documents are selected.

    ### 6. **Generating the Final Answer:**
    - The retrieved documents are added as **context** to the prompt that is sent to the Large Language Model (LLM). The documents provide background information that augments the answer generation.
    - The LLM (using the HuggingFace pipeline) generates a response based on the provided **question** and **context**.
    - The prompt template ensures that the model knows what context to use and how to format the answer.
    - The final response is returned and presented to the user.

    ### 7. **Answer and Context Generation Flow:**
    - **Question**: The recruiter’s query (e.g., “Tell me about your experience with computer vision”).
    - **Context**: Relevant documents retrieved from the vector database and reranked for relevance.
    - **LLM Response**: The LLM, using both the question and augmented context, generates a professional, concise answer.

    ### Example Flow:
    1. A recruiter asks a question about a candidate's experience with **AI**.
    2. The system retrieves the most relevant documents from the database using **FAISS** based on vector similarity.
    3. The retrieved documents are reranked for relevance.
    4. These documents, along with the question, are passed to the LLM for generating an answer.
    5. The LLM generates a clear, tailored response that highlights the most relevant experiences and skills of the candidate.

    ### Key Benefits of This System:
    - **Enhanced Relevance**: By retrieving documents related to the question, the system ensures that the answers are relevant and based on real-time data.
    - **Scalability**: As the dataset grows (more documents are added), the retrieval process ensures that answers can scale with more knowledge.
    - **Efficiency**: FAISS allows for fast similarity search, enabling quick responses to recruiters’ queries.
    - **Adaptability**: The system can be trained and adapted with new candidate data, making it useful for continuous hiring processes.

    This approach combines powerful information retrieval with language generation, creating a robust system for answering detailed, context-aware questions in recruitment scenarios.
    """)

# Creating an expander for "Limitations"
with st.expander("Limitations"):
    st.write("""
        1. **Single Instance**: Only one instance of the system is available at a time.
        2. **Idle Timeout**: The system will reset after **5 minutes** of inactivity.
        3. **Resource Limitations**: This service is hosted using **ngrok** (free version) for tunneling, which might limit the duration and access to the service, you may experience some delays or resource limitations.
    """)

# System Initialization Section
if not st.session_state["initialized"]:
    st.button("Initialize System", on_click=initialize_system)

# Main Functionality
if st.session_state["initialized"]:
    st.header("Ask a Question")

    # Check if 'question_input' is properly initialized
    if "question_input" not in st.session_state or not st.session_state["question_input"]:
        st.session_state["question_input"] = ""  # Ensure it's an empty string

    # Question Input and Submit Button
    question = st.text_area("Enter your question:", value=st.session_state["question_input"], height=200)

    if st.button("Submit"):
        if question.strip():  # Ensure a non-empty input
            with st.spinner("Retrieving answer..."):
                try:
                    # Retrieve answer using RAG
                    answer, relevant_docs = answer_with_rag(question, st.session_state["llm"], st.session_state["vector_db"])

                    # Display the answer
                    st.subheader("Answer")
                    st.write(answer)

                    # Log the interaction
                    add_log_with_location(question, answer)
                    add_log(f"Answered question: '{question}'")

                    # Clear the input field for the next question
                    st.session_state["question_input"] = ""
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
        else:
            st.warning("Please enter a question before submitting.")


    # Deinitialize Button
    st.button("Deinitialize System", on_click=deinitialize_system)