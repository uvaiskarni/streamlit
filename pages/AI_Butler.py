import torch
import gc
import os
import time
import streamlit as st
from transformers import AutoTokenizer
from lib.rag_libs import initialise_llm, answer_with_rag
from lib.rag_libs import load_vector_database, generate_save_vector_database, clear_memory

# Initialize session state for logs if not already initialized
if "logs" not in st.session_state:
    st.session_state.logs = []

# Function to add log messages
def add_log(message):
    # Ensure the logs key exists before appending
    if "logs" not in st.session_state:
        st.session_state.logs = []
    st.session_state.logs.append(f"[{time.strftime('%H:%M:%S')}] {message}")
    # Limit the logs to the last 50 entries
    if len(st.session_state.logs) > 50:  # Keep logs manageable
        st.session_state.logs.pop(0)

# Example of calling add_log function
add_log("System started successfully.")

# Example of shutdown function that adds a log message
def shutdown():
    add_log("System shutdown complete. Exiting application.")
    st.stop()

# Constants
CACHE_PATH = r"C:\Users\Uvais\Documents\coding\streamlit\cache"
DATA_DIR = r"C:\Users\Uvais\Documents\coding\streamlit\docs"
IDLE_TIMEOUT = 300  # 5 minutes in seconds

# Idle Timer State
if "last_activity" not in st.session_state:
    st.session_state.last_activity = time.time()

# Load or Generate Vector Database
def load_or_generate_vector_db():
    if (
        os.path.exists(os.path.join(CACHE_PATH, "vector_db.index")) and
        os.path.exists(os.path.join(CACHE_PATH, "docstore.pkl")) and
        os.path.exists(os.path.join(CACHE_PATH, "index_to_docstore_id.pkl"))
    ):
        add_log("Loading cached vector database...")
        vector_db = load_vector_database(CACHE_PATH)
        add_log("Cached vector database loaded.")
    else:
        add_log("Generating new vector database...")
        vector_db = generate_save_vector_database(DATA_DIR)
        add_log("New vector database generated and saved.")
    return vector_db

# GPU Reset & Memory Cleanup Function
def reset_gpu_and_clear_memory():
    add_log("Releasing GPU memory and clearing cache...")
    
    # Reset GPU memory (only if using CUDA)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        add_log("GPU memory cleared.")
    
    # Clear Python memory and garbage collection
    gc.collect()
    add_log("Python memory cleared.")

# Shutdown Functionality
def shutdown():
    add_log("Shutting down system...")

    # Release models and data
    clear_memory()
    add_log("Model memory cleared.")
    
    # Reset GPU resources and clear cache
    reset_gpu_and_clear_memory()
    
    # Clear session state
    st.session_state.clear()

    add_log("System shutdown complete. Exiting application.")
    st.stop()

# Page Config
st.set_page_config(page_title="AI Butler", layout="centered")

# App Title and Disclaimer
st.title("AI Butler")

st.subheader("AI Assistant to learn more about candidate (Myself)")

# Creating an expander for "How to use"
with st.expander("How to use"):
    st.write("""
        1. To use the application, first intialise code by clicking on button "Initialise System".
        2. You enter your questions in the Ask a question text box (Press Ctrl+Enter to generate answer. or tap on the box right side if on a touch interface)     
        3. The genrated will be displayed on the same page.
        4. You can view the logs by expanding the side tab section.
        5. Deintialize the resources using button "Deinitialise System"- 
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
        2. **Requires Reinitialization**: The system may need to be reinitialized after deinitialization due to session state persistence or other application state factors (i.e click deintialise twice).
        3. **Idle Timeout**: The system will reset after **5 minutes** of inactivity.
        4. **Resource Limitations**: This service is hosted using **ngrok** (free version) for tunneling, which might limit the duration and access to the service, you may experience some delays or resource limitations.
    """)

# Sidebar for Terminal Logs
with st.sidebar:
    st.subheader("Terminal Logs")
    for log in st.session_state.logs:
        st.text(log)

# Check for Idle Timeout
if time.time() - st.session_state.last_activity > IDLE_TIMEOUT:
    add_log("Idle timeout reached. Shutting down application.")
    shutdown()

# Refresh Activity Timer
st.session_state.last_activity = time.time()

# Initialization
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
    st.session_state.llm = None
    st.session_state.initialised = False

# Initialize Button
if not st.session_state.initialised:
    if st.button("Initialize System"):
        with st.spinner("Initializing..."):
            st.session_state.vector_db = load_or_generate_vector_db()
            st.session_state.llm = initialise_llm()
            st.session_state.initialised = True
            add_log("System initialized successfully.")

# Chat Interface
if st.session_state.initialised:
    question = st.text_area("Ask a question:", key="chat_input", height=200)
    if question:
        st.session_state.last_activity = time.time()
        with st.spinner("Retrieving answer..."):
            answer, relevant_docs = answer_with_rag(
                question,
                st.session_state.llm,
                st.session_state.vector_db,
            )
            st.subheader("Answer")
            st.write(f"**AI Butler Response:** {answer}")
            add_log(f"Answered question: '{question}'")


# Reset Button
if st.session_state.initialised:
    if st.button("Deinitialize System"):
        # Clear all session state related to the system and conversation
        reset_gpu_and_clear_memory()
        add_log("GPU and memory cleared.")
        
        # Reset variables
        st.session_state.vector_db = None
        st.session_state.llm = None
        st.session_state.initialised = False
        
        # Clear logs and any conversation context
        st.session_state.logs = []
        add_log("System deinitialized, resources reset.")
