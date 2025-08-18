# academic_deep_search.py

import streamlit as st
import os
import numpy as np
import json
from sklearn.cluster import KMeans
from typing import List, Optional, Dict, Any, TypedDict, Annotated, Tuple
import operator
import uuid
from fpdf import FPDF
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

# --- CONFIGURATION ---

# Default academic domains to search
DEFAULT_DOMAINS = ["ieeexplore.ieee.org", "dl.acm.org", "arxiv.org", "scholar.google.com"]
GEMINI_MODELS = ["gemini-2.5-pro", "gemini-2.5-flash"]

# --- RAPTOR IMPLEMENTATION ---

class RAPTORRetriever(BaseRetriever):
    """A custom retriever that wraps the RAPTOR index for LangChain compatibility."""
    raptor_index: Any
    
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        return self.raptor_index.retrieve(query)

class RAPTOR:
    """
    RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval
    This class implements the RAPTOR indexing and retrieval mechanism with checkpointing.
    """
    def __init__(self, llm, embeddings_model, session_id, chunk_size=1000, chunk_overlap=200):
        self.llm = llm
        self.embeddings_model = embeddings_model
        self.session_id = session_id
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.tree = {}
        self.all_nodes = {}
        self.vector_store = None
        self.node_ids = []
        self.checkpoint_path = f"checkpoint_{self.session_id}.json"

    def _save_checkpoint(self, level):
        """Saves the current state of the tree to a file."""
        state = {
            "level": level,
            "tree": {str(k): v for k, v in self.tree.items()}, # Ensure keys are strings
            "all_nodes": self.all_nodes,
        }
        with open(self.checkpoint_path, 'w') as f:
            json.dump(state, f)
        st.write(f"Checkpoint saved for level {level}.")

    def _load_checkpoint(self) -> int:
        """Loads the state from a checkpoint file if it exists."""
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, 'r') as f:
                    state = json.load(f)
                self.tree = state["tree"]
                self.all_nodes = state["all_nodes"]
                start_level = state["level"]
                st.info(f"Resuming from checkpoint at level {start_level}.")
                return start_level
            except (json.JSONDecodeError, KeyError) as e:
                st.warning(f"Could not load checkpoint file due to error: {e}. Starting from scratch.")
                return 0
        return 0

    def add_documents(self, text: str):
        start_level = self._load_checkpoint()
        
        if start_level == 0:
            st.write("Step 1: Splitting text into initial chunks (Level 0)...")
            initial_chunks = self.text_splitter.create_documents([text])
            level_0_texts = [doc.page_content for doc in initial_chunks]
            
            for i, chunk_text in enumerate(level_0_texts):
                self.all_nodes[f"0_{i}"] = chunk_text
            
            self.tree[str(0)] = level_0_texts
            self._save_checkpoint(0) # Save initial state
        
        current_level = start_level
        
        while len(self.tree[str(current_level)]) > 1:
            next_level = current_level + 1
            st.write(f"Step 2: Building Level {next_level} of the tree...")
            
            current_level_nodes = self.tree[str(current_level)]
            clustered_indices = self._cluster_nodes(current_level_nodes)
            
            next_level_nodes = []
            num_clusters = len(clustered_indices)
            summary_progress = st.progress(0, text=f"Summarizing Level {next_level}...")
            
            for i, indices in enumerate(clustered_indices):
                summary = self._summarize_cluster([current_level_nodes[j] for j in indices])
                next_level_nodes.append(summary)
                self.all_nodes[f"{next_level}_{i}"] = summary
                summary_progress.progress((i + 1) / num_clusters, text=f"Summarizing cluster {i+1}/{num_clusters} for Level {next_level}...")
            
            self.tree[str(next_level)] = next_level_nodes
            self._save_checkpoint(next_level)
            current_level = next_level

        st.write("Step 3: Creating final vector store from all nodes...")
        self.node_ids = list(self.all_nodes.keys())
        node_texts = list(self.all_nodes.values())
        
        self.vector_store = FAISS.from_texts(texts=node_texts, embedding=self.embeddings_model)
        st.write("RAPTOR index built successfully!")

        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)

    def _cluster_nodes(self, node_texts: List[str]) -> List[List[int]]:
        st.write(f"Embedding {len(node_texts)} nodes for clustering...")
        embeddings = self.embeddings_model.embed_documents(node_texts)
        
        n_clusters = max(2, len(node_texts) // 10)
        
        st.write(f"Clustering into {n_clusters} groups...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(embeddings)
        
        clusters = [[] for _ in range(n_clusters)]
        for i, label in enumerate(kmeans.labels_):
            clusters[label].append(i)
            
        return clusters

    def _summarize_cluster(self, cluster_texts: List[str]) -> str:
        context = "\n\n---\n\n".join(cluster_texts)
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are an AI assistant that summarizes academic texts. Create a concise, abstractive summary of the following content, synthesizing the key information."),
            HumanMessage(content=f"Please summarize the following content:\n\n{context}")
        ])
        
        response = self.llm.invoke(prompt)
        return response.content

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        return self.vector_store.similarity_search(query, k=k) if self.vector_store else []
    
    def as_retriever(self) -> BaseRetriever:
        return RAPTORRetriever(raptor_index=self)

# --- STATE DEFINITION FOR LANGGRAPH ---
class ResearchState(TypedDict):
    query: str
    domains: List[str]
    num_references_per_domain: int
    paper_urls: Annotated[List[str], operator.add]
    downloaded_pdf_paths: Annotated[List[str], operator.add]
    extracted_text: str
    raptor_index: Any
    conversation_history: Annotated[List[BaseMessage], operator.add]
    generation: str

# --- LANGGRAPH NODES AND GRAPH DEFINITION ---
def start_search_node(state: ResearchState) -> ResearchState:
    st.write("Starting research process...")
    return {"paper_urls": [], "downloaded_pdf_paths": [], "extracted_text": ""}

def web_search_node(state: ResearchState) -> ResearchState:
    st.write("Searching for academic papers...")
    all_urls = []
    for domain in state["domains"]:
        try:
            from duckduckgo_search import DDGS
            query = f'site:{domain} {state["query"]} filetype:pdf'
            with DDGS() as ddgs:
                results = [r['href'] for r in ddgs.text(query, max_results=state["num_references_per_domain"])]
                st.write(f"Found {len(results)} papers on {domain}.")
                all_urls.extend(results)
        except Exception as e:
            st.warning(f"Could not search {domain}: {e}")
    unique_urls = list(set(all_urls))
    st.write(f"Found a total of {len(unique_urls)} unique papers.")
    return {"paper_urls": unique_urls}

def download_pdfs_node(state: ResearchState) -> ResearchState:
    st.write("Downloading PDFs...")
    pdf_paths = []
    if not os.path.exists("temp_pdfs"):
        os.makedirs("temp_pdfs")
    
    total_urls = len(state["paper_urls"])
    if total_urls == 0:
        st.warning("No paper URLs found to download.")
        return {"downloaded_pdf_paths": []}
        
    download_progress = st.progress(0, text="Starting download...")

    for i, url in enumerate(state["paper_urls"]):
        download_progress.progress((i + 1) / total_urls, text=f"Downloading paper {i+1}/{total_urls}...")
        try:
            response = requests.get(url, timeout=20, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            filename = f"temp_pdfs/{uuid.uuid4()}.pdf"
            with open(filename, "wb") as f:
                f.write(response.content)
            pdf_paths.append(filename)
        except Exception as e:
            st.warning(f"Failed to download {url}: {e}")
            
    return {"downloaded_pdf_paths": pdf_paths}

def extract_text_node(state: ResearchState) -> ResearchState:
    st.write("Extracting text from PDFs...")
    full_text = ""
    for path in state["downloaded_pdf_paths"]:
        try:
            loader = PyPDFLoader(path)
            pages = loader.load_and_split()
            full_text += "\n\n".join([page.page_content for page in pages])
        except Exception as e:
            st.warning(f"Could not read {path}: {e}")
    st.write(f"Extracted a total of {len(full_text)} characters.")
    return {"extracted_text": full_text}

def get_llm_and_embeddings(provider: str, model_name: str, embeddings_model_name: Optional[str] = None):
    if provider == "gemini":
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.3, convert_system_message_to_human=True)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    else: # Ollama
        llm = ChatOllama(model=model_name, temperature=0.3)
        embed_model = embeddings_model_name if embeddings_model_name else model_name
        embeddings = OllamaEmbeddings(model=embed_model)
    return llm, embeddings

def build_raptor_index_node(state: ResearchState) -> ResearchState:
    st.write("Building RAPTOR index... This may take some time.")
    model_config = st.session_state.get("model_config", {})
    provider = model_config.get("provider")
    model_name = model_config.get("model_name")
    embeddings_model_name = model_config.get("embeddings_model_name")
    
    if not provider or not model_name:
        st.error("LLM provider or model not configured correctly.")
        return {"raptor_index": None}

    llm, embeddings = get_llm_and_embeddings(provider, model_name, embeddings_model_name=embeddings_model_name)
    
    if not state["extracted_text"]:
        st.error("No text was extracted from PDFs. Cannot build index.")
        return {"raptor_index": None}

    raptor_index = RAPTOR(
        llm=llm, 
        embeddings_model=embeddings, 
        session_id=st.session_state.session_id
    )
    raptor_index.add_documents(state["extracted_text"])
    
    st.success("Research and indexing complete! You can now ask questions.")
    return {"raptor_index": raptor_index}

builder = StateGraph(ResearchState)
builder.add_node("start_search", start_search_node)
builder.add_node("web_search", web_search_node)
builder.add_node("download_pdfs", download_pdfs_node)
builder.add_node("extract_text", extract_text_node)
builder.add_node("build_raptor_index", build_raptor_index_node)
builder.add_edge(START, "start_search")
builder.add_edge("start_search", "web_search")
builder.add_edge("web_search", "download_pdfs")
builder.add_edge("download_pdfs", "extract_text")
builder.add_edge("extract_text", "build_raptor_index")
builder.add_edge("build_raptor_index", END)
graph = builder.compile()

# --- HELPER FUNCTIONS FOR EXPORT ---
def generate_pdf_report(chat_history: List[Dict[str, str]]) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Academic QA Chat History", ln=True, align='C')
    for message in chat_history:
        role = message.get('role', '')
        content = message.get('content', '')
        if role == 'user':
            pdf.set_text_color(0, 0, 255)
            pdf.multi_cell(0, 10, f"Question: {content}")
        else:
            pdf.set_text_color(0, 0, 0)
            pdf.multi_cell(0, 10, f"Answer: {content}")
        pdf.ln(5)
    return pdf.output(dest='S').encode('latin-1')

def generate_mermaid_diagram(query: str, domains: List[str], num_refs: int, found_urls: int) -> str:
    return f"""graph TD; A[Start: User Input] --> B(LangGraph Pipeline); B --> C[Web Search]; C --> D[Download PDFs]; D --> E[Extract Text]; E --> F[Build RAPTOR Index]; F --> G[Conversational QA]; subgraph Parameters; P1("Query: {query}"); P2("Domains: {', '.join(domains)}"); P3("References per Domain: {num_refs}"); end; subgraph Results; R1("Found URLs: {found_urls}"); end; A -- Parameters --> P1 & P2 & P3; C -- Results --> R1;"""

# --- STREAMLIT UI ---
@st.cache_data(show_spinner=False)
def get_ollama_models():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        response.raise_for_status()
        return [model['name'] for model in response.json().get('models', [])]
    except (requests.exceptions.RequestException, KeyError):
        return []

def main():
    st.set_page_config(layout="wide", page_title="Academic Deep Search")
    st.title("📚 Academic Deep Search & QA with RAPTOR")

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.research_done = False
        st.session_state.final_state = None
        st.session_state.model_config = {}

    with st.sidebar:
        st.header("1. Research Parameters")
        query = st.text_input("Academic Topic", "indoor quality monitoring using machine learning")
        domains_str = st.text_area("Webpage Domains (one per line)", "\n".join(DEFAULT_DOMAINS))
        num_references = st.slider("References per Domain", 1, 50, 5)

        st.header("2. AI Model Configuration")
        llm_provider = st.selectbox("LLM Provider", ["Ollama", "Gemini"], key="llm_provider")

        model_name = None
        embeddings_model_name = None

        if llm_provider == "Ollama":
            ollama_models = get_ollama_models()
            if ollama_models:
                # Suggest a common chat model as default if available.
                default_chat_index = ollama_models.index("llama3:8b") if "llama3:8b" in ollama_models else 0
                model_name = st.selectbox("Select a Chat Model", ollama_models, index=default_chat_index)
                
                # Suggest a common embeddings model as default if available.
                default_embed_index = ollama_models.index("mxbai-embed-large") if "mxbai-embed-large" in ollama_models else 0
                embeddings_model_name = st.selectbox("Select an Embeddings Model", ollama_models, index=default_embed_index)
            else:
                st.warning("Ollama not detected. Please enter model names manually.")
                model_name = st.text_input("Ollama Chat Model Name", "llama3")
                embeddings_model_name = st.text_input("Ollama Embeddings Model Name", "mxbai-embed-large")
        
        elif llm_provider == "Gemini":
            google_api_key = os.environ.get("GOOGLE_API_KEY")
            if not google_api_key:
                google_api_key = st.text_input("Enter your Google API Key", type="password")
            if google_api_key:
                os.environ["GOOGLE_API_KEY"] = google_api_key
                model_name = st.selectbox("Select a Gemini Model", GEMINI_MODELS)
            else:
                st.warning("Google API Key is required to use Gemini.")
        
        if model_name:
            st.session_state.model_config = {
                "provider": llm_provider.lower(),
                "model_name": model_name,
                "embeddings_model_name": embeddings_model_name
            }
        
        st.header("3. Start Research")
        if st.button("Start Research Pipeline"):
            if not st.session_state.model_config.get("model_name"):
                st.error("Please configure the AI model before starting.")
            else:
                st.session_state.research_done = False
                st.session_state.messages = []
                
                # Clear any old checkpoints from previous runs, but not for the current session
                # This button click signifies a new research task
                checkpoint_file = f"checkpoint_{st.session_state.session_id}.json"
                if os.path.exists(checkpoint_file):
                    os.remove(checkpoint_file)

                with st.spinner("Running deep research pipeline..."):
                    domains = [d.strip() for d in domains_str.split("\n") if d.strip()]
                    initial_state = {"query": query, "domains": domains, "num_references_per_domain": num_references}
                    final_state = graph.invoke(initial_state)
                    if final_state.get("raptor_index"):
                        st.session_state.research_done = True
                        st.session_state.final_state = final_state
                    else:
                        st.error("Research pipeline failed to build an index. Check logs.")

    if st.session_state.research_done:
        st.header("Conversational QA")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask a question about the papers..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    retriever = st.session_state.final_state['raptor_index'].as_retriever()
                    model_config = st.session_state.model_config
                    llm, _ = get_llm_and_embeddings(
                        provider=model_config["provider"],
                        model_name=model_config["model_name"],
                        embeddings_model_name=model_config.get("embeddings_model_name")
                    )
                    
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", "You are an AI research assistant. Answer based on the following context from academic papers:\n\n{context}\n\nIf the answer isn't in the context, say so."),
                        ("human", "{question}")
                    ])
                    
                    retrieved_docs = retriever.get_relevant_documents(prompt)
                    context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
                    
                    chain = prompt_template | llm
                    response = chain.invoke({"context": context, "question": prompt})
                    response_content = response.content
                    st.markdown(response_content)
            
            st.session_state.messages.append({"role": "assistant", "content": response_content})
        
        with st.expander("Export Options"):
            if st.button("Export Chat as PDF"):
                pdf_bytes = generate_pdf_report(st.session_state.messages)
                st.download_button(label="Download PDF", data=pdf_bytes, file_name="chat_history.pdf", mime="application/pdf")

            if st.button("Generate Pipeline Diagram"):
                final_state = st.session_state.final_state
                mermaid_code = generate_mermaid_diagram(
                    query=final_state['query'], domains=final_state['domains'],
                    num_refs=final_state['num_references_per_domain'], found_urls=len(final_state['paper_urls']))
                st.code(mermaid_code, language="mermaid")
                st.info("Copy this code and render it in a Mermaid.js compatible viewer.")
    else:
        st.info("Configure your research and AI model in the sidebar, then click 'Start Research Pipeline'.")

if __name__ == "__main__":
    main()