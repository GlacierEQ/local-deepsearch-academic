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
from bs4 import BeautifulSoup
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

# --- CONFIGURATION ---

DEFAULT_DOMAINS = ["arxiv.org", "ieeexplore.ieee.org", "dl.acm.org"]

# --- RAPTOR IMPLEMENTATION (No changes here) ---

class RAPTORRetriever(BaseRetriever):
    raptor_index: Any
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        return self.raptor_index.retrieve(query)

class RAPTOR:
    def __init__(self, llm, embeddings_model, session_id, chunk_size=1000, chunk_overlap=200):
        self.llm = llm
        self.embeddings_model = embeddings_model
        self.session_id = session_id
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.tree = {}
        self.all_nodes: Dict[str, Document] = {}
        self.vector_store = None
        self.checkpoint_path = f"checkpoint_{self.session_id}.json"

    def _save_checkpoint(self, level):
        state = {
            "level": level,
            "tree": {str(k): [node_id for node_id in v] for k, v in self.tree.items()},
            "all_nodes": {node_id: doc.to_json() for node_id, doc in self.all_nodes.items()},
        }
        with open(self.checkpoint_path, 'w') as f:
            json.dump(state, f)
        st.write(f"Checkpoint saved for level {level}.")

    def _load_checkpoint(self) -> int:
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, 'r') as f:
                    state = json.load(f)
                from langchain_core.load import load
                self.all_nodes = {node_id: load(doc) for node_id, doc in state["all_nodes"].items()}
                self.tree = state["tree"]
                start_level = state["level"]
                st.info(f"Resuming from checkpoint at level {start_level}.")
                return start_level
            except Exception as e:
                st.warning(f"Could not load checkpoint file due to error: {e}. Starting from scratch.")
                return 0
        return 0

    def add_documents(self, documents: List[Document]):
        start_level = self._load_checkpoint()
        if start_level == 0:
            st.write("Step 1: Assigning IDs to initial chunks (Level 0)...")
            level_0_node_ids = []
            for i, doc in enumerate(documents):
                node_id = f"0_{i}"
                self.all_nodes[node_id] = doc
                level_0_node_ids.append(node_id)
            self.tree[str(0)] = level_0_node_ids
            self._save_checkpoint(0)
        
        current_level = start_level
        while len(self.tree[str(current_level)]) > 1:
            next_level = current_level + 1
            st.write(f"Step 2: Building Level {next_level} of the tree...")
            current_level_node_ids = self.tree[str(current_level)]
            current_level_docs = [self.all_nodes[nid] for nid in current_level_node_ids]
            clustered_indices = self._cluster_nodes(current_level_docs)
            
            next_level_node_ids = []
            num_clusters = len(clustered_indices)
            summary_progress = st.progress(0, text=f"Summarizing Level {next_level}...")
            for i, indices in enumerate(clustered_indices):
                cluster_docs = [current_level_docs[j] for j in indices]
                summary, combined_metadata = self._summarize_cluster(cluster_docs)
                summary_doc = Document(page_content=summary, metadata=combined_metadata)
                node_id = f"{next_level}_{i}"
                self.all_nodes[node_id] = summary_doc
                next_level_node_ids.append(node_id)
                summary_progress.progress((i + 1) / num_clusters, text=f"Summarizing cluster {i+1}/{num_clusters} for Level {next_level}...")
            
            self.tree[str(next_level)] = next_level_node_ids
            self._save_checkpoint(next_level)
            current_level = next_level

        st.write("Step 3: Creating final vector store from all nodes...")
        final_docs = list(self.all_nodes.values())
        self.vector_store = FAISS.from_documents(documents=final_docs, embedding=self.embeddings_model)
        st.write("RAPTOR index built successfully!")
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)

    def _cluster_nodes(self, docs: List[Document]) -> List[List[int]]:
        st.write(f"Embedding {len(docs)} nodes for clustering...")
        embeddings = self.embeddings_model.embed_documents([doc.page_content for doc in docs])
        n_clusters = max(2, len(docs) // 10)
        st.write(f"Clustering into {n_clusters} groups...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(embeddings)
        clusters = [[] for _ in range(n_clusters)]
        for i, label in enumerate(kmeans.labels_):
            clusters[label].append(i)
        return clusters

    def _summarize_cluster(self, cluster_docs: List[Document]) -> Tuple[str, dict]:
        context = "\n\n---\n\n".join([doc.page_content for doc in cluster_docs])
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are an AI assistant that summarizes academic texts. Create a concise, abstractive summary of the following content, synthesizing the key information."),
            HumanMessage(content=f"Please summarize the following content:\n\n{context}")
        ])
        response = self.llm.invoke(prompt)
        summary = response.content
        aggregated_sources = list(set(doc.metadata.get("url", "Unknown Source") for doc in cluster_docs))
        combined_metadata = {"sources": aggregated_sources}
        return summary, combined_metadata

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        return self.vector_store.similarity_search(query, k=k) if self.vector_store else []
    
    def as_retriever(self) -> BaseRetriever:
        return RAPTORRetriever(raptor_index=self)

# --- STATE DEFINITION FOR LANGGRAPH ---
class ResearchState(TypedDict):
    query: str
    domains: List[str]
    num_references_per_domain: int
    start_year: int
    end_year: int
    landing_page_urls: Annotated[List[str], operator.add]
    direct_pdf_urls: Annotated[List[str], operator.add]
    path_to_url_map: dict
    extracted_docs: list
    raptor_index: Any
    conversation_history: Annotated[List[BaseMessage], operator.add]
    generation: str

# --- LANGGRAPH NODES AND GRAPH DEFINITION ---
def start_search_node(state: ResearchState) -> ResearchState:
    st.write("Starting research process...")
    return {"landing_page_urls": [], "direct_pdf_urls": [], "path_to_url_map": {}, "extracted_docs": []}

def web_search_node(state: ResearchState) -> ResearchState:
    st.write("Stage 1: Finding article landing pages...")
    all_landing_urls = []
    
    for domain in state["domains"]:
        domain_urls = []
        st.write(f"-> Using web search for '{domain}'")
        try:
            from tavily import TavilyClient

            tavily = TavilyClient(api_key=os.get_environ("TAVILY_API_KEY"))

            num_to_fetch = 100000

            response = tavily.search(
                query=state["query"],
                search_depth="advanced",
                include_answer=True,
                max_results=num_to_fetch,
                include_domains=[domain]
            )
            
            # Fetch a larger pool of results to filter through
            
            domain_urls = [r['url'] for r in response]                # Filter for valid URLs and stop when we have enough 
           
        except Exception as e:
            st.warning(f"Could not search {domain}: {e}")
        
        all_landing_urls.extend(domain_urls)
                
    unique_urls = list(set(all_landing_urls))
    st.write(f"Found a total of {len(unique_urls)} unique article pages across all domains.")
    return {"landing_page_urls": unique_urls}

def find_pdf_links_node(state: ResearchState) -> ResearchState:
    st.write("Stage 2: Scraping landing pages for direct PDF links...")
    direct_urls = []
    
    total_urls = len(state["landing_page_urls"])
    if total_urls == 0:
        st.warning("No landing pages found to scrape.")
        return {"direct_pdf_urls": []}
        
    scrape_progress = st.progress(0, text="Starting scraping...")
    
    for i, url in enumerate(state["landing_page_urls"]):
        scrape_progress.progress((i + 1) / total_urls, text=f"Scraping page {i+1}/{total_urls}...")
        try:
            response = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Heuristic to find PDF links
            pdf_link = soup.find('a', href=lambda href: href and href.lower().endswith('.pdf'))
            
            if pdf_link and pdf_link.get('href'):
                pdf_url = pdf_link['href']
                if not pdf_url.startswith('http'):
                    from urllib.parse import urljoin
                    pdf_url = urljoin(url, pdf_url)
                direct_urls.append(pdf_url)
        except Exception as e:
            st.warning(f"Could not scrape {url} for PDF link: {e}")

    unique_urls = list(set(direct_urls))
    st.write(f"Found {len(unique_urls)} direct PDF links.")
    return {"direct_pdf_urls": unique_urls}

def download_pdfs_node(state: ResearchState) -> ResearchState:
    st.write("Stage 3: Downloading valid PDFs...")
    path_to_url_map = {}
    if not os.path.exists("temp_pdfs"):
        os.makedirs("temp_pdfs")
    
    total_urls = len(state["direct_pdf_urls"])
    if total_urls == 0:
        st.warning("No direct PDF URLs found to download.")
        return {"path_to_url_map": {}}
        
    download_progress = st.progress(0, text="Starting download...")

    for i, url in enumerate(state["direct_pdf_urls"]):
        download_progress.progress((i + 1) / total_urls, text=f"Downloading paper {i+1}/{total_urls}...")
        try:
            response = requests.get(url, timeout=20, headers={'User-Agent': 'Mozilla/5.0'}, stream=True)
            response.raise_for_status()
            content_type = response.headers.get('Content-Type', '')
            if 'application/pdf' in content_type.lower():
                filename = f"temp_pdfs/{uuid.uuid4()}.pdf"
                with open(filename, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                path_to_url_map[filename] = url
            else:
                st.warning(f"Skipped non-PDF file at {url} (Content-Type: {content_type})")
        except requests.exceptions.RequestException as e:
            st.warning(f"Failed to download {url}: {e}")
            
    return {"path_to_url_map": path_to_url_map}

def extract_text_node(state: ResearchState) -> ResearchState:
    st.write("Stage 4: Extracting text from PDFs...")
    all_docs = []
    path_to_url_map = state["path_to_url_map"]
    for path, url in path_to_url_map.items():
        try:
            loader = PyPDFLoader(path)
            pages = loader.load_and_split()
            for page in pages:
                page.metadata["url"] = url
            all_docs.extend(pages)
        except Exception as e:
            st.warning(f"Could not read or parse PDF from {url}. It may be corrupted. Error: {e}")
            
    st.write(f"Successfully extracted {len(all_docs)} document chunks.")
    return {"extracted_docs": all_docs}

def get_llm_and_embeddings(model_name: str, embeddings_model_name: Optional[str] = None):
    llm = ChatOllama(model=model_name, temperature=0.3)
    embed_model = embeddings_model_name if embeddings_model_name else model_name
    embeddings = OllamaEmbeddings(model=embed_model)
    return llm, embeddings

def build_raptor_index_node(state: ResearchState) -> ResearchState:
    st.write("Stage 5: Building RAPTOR index... This may take some time.")
    model_config = st.session_state.get("model_config", {})
    chat_model_name = model_config.get("model_name")
    summary_model_name = model_config.get("summary_model_name")
    embeddings_model_name = model_config.get("embeddings_model_name")
    
    if not chat_model_name:
        st.error("Ollama chat model not configured correctly.")
        return {"raptor_index": None}

    summarizer_model_name = summary_model_name if summary_model_name else chat_model_name
    llm_for_summaries = ChatOllama(model=summarizer_model_name, temperature=0.3)
    
    embed_model_name = embeddings_model_name if embeddings_model_name else chat_model_name
    embeddings = OllamaEmbeddings(model=embed_model_name)
    
    if not state["extracted_docs"]:
        st.error("No valid documents were extracted. Cannot build index.")
        return {"raptor_index": None}

    raptor_index = RAPTOR(
        llm=llm_for_summaries, 
        embeddings_model=embeddings, 
        session_id=st.session_state.session_id
    )
    raptor_index.add_documents(state["extracted_docs"])
    
    st.success("Research and indexing complete! You can now ask questions.")
    return {"raptor_index": raptor_index}

builder = StateGraph(ResearchState)
builder.add_node("start_search", start_search_node)
builder.add_node("web_search", web_search_node)
builder.add_node("find_pdf_links", find_pdf_links_node)
builder.add_node("download_pdfs", download_pdfs_node)
builder.add_node("extract_text", extract_text_node)
builder.add_node("build_raptor_index", build_raptor_index_node)
builder.add_edge(START, "start_search")
builder.add_edge("start_search", "web_search")
builder.add_edge("web_search", "find_pdf_links")
builder.add_edge("find_pdf_links", "download_pdfs")
builder.add_edge("download_pdfs", "extract_text")
builder.add_edge("extract_text", "build_raptor_index")
builder.add_edge("build_raptor_index", END)
graph = builder.compile()

# --- HELPER FUNCTIONS & UI ---
def generate_pdf_report(chat_history: List[Dict[str, str]], used_sources: List[str]) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', size=16)
    pdf.cell(0, 10, txt="Academic Q&A Chat History", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    for message in chat_history:
        role, content = message.get('role', ''), message.get('content', '')
        if role == 'user':
            pdf.set_font('Arial', 'B', 12)
            pdf.set_text_color(0, 0, 128)
            pdf.multi_cell(0, 10, f"Question: {content}")
        else:
            pdf.set_font('Arial', '', 12)
            pdf.set_text_color(0, 0, 0)
            pdf.multi_cell(0, 10, f"Answer: {content}")
        pdf.ln(5)
    if used_sources:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, txt="References", ln=True, align='L')
        pdf.ln(5)
        pdf.set_font("Arial", size=10)
        for i, source in enumerate(used_sources):
            pdf.multi_cell(0, 8, f"{i+1}. {source}")
    return pdf.output(dest='S').encode('latin-1')

def generate_bibliography_pdf(all_sources: List[str]) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', size=16)
    pdf.cell(0, 10, txt="Full Bibliography of Scraped Articles", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=10)
    for i, source in enumerate(all_sources):
        pdf.multi_cell(0, 8, f"{i+1}. {source}")
    return pdf.output(dest='S').encode('latin-1')

def generate_mermaid_diagram(query: str, domains: List[str], num_refs: int, found_urls: int, start_year: int, end_year: int) -> str:
    return f"""graph TD; A[Start: User Input] --> B(LangGraph Pipeline); B --> C[Web Search for Landing Pages]; C --> D[Scrape for PDF Links]; D --> E[Download & Validate PDFs]; E --> F[Extract Text]; F --> G[Build RAPTOR Index]; G --> H[Conversational QA]; subgraph Parameters; P1("Query: {query}"); P2("Domains: {', '.join(domains)}"); P3("References per Domain: {num_refs}"); P4("Years: {start_year}-{end_year}"); end; subgraph Results; R1("Found URLs: {found_urls}"); end; C -- found pages --> D; D -- found links --> E; """

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
    st.markdown("Powered by Ollama 🦙")

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.research_done = False
        st.session_state.final_state = None
        st.session_state.model_config = {}
        st.session_state.used_sources = set()

    with st.sidebar:
        st.header("1. Research Parameters")
        query = st.text_input("Academic Topic", "indoor quality monitoring using machine learning")
        domains_str = st.text_area("Webpage Domains (one per line)", "\n".join(DEFAULT_DOMAINS))
        num_references = st.slider("References per Domain", 1, 50, 5)
        
        current_year = datetime.now().year
        start_year = st.number_input("Start Year", min_value=1980, max_value=current_year, value=2020)
        end_year = st.number_input("End Year", min_value=1980, max_value=current_year, value=current_year)
        
        if start_year > end_year:
            st.error("Error: Start year cannot be after end year.")

        st.header("2. AI Model Configuration")
        
        ollama_models = get_ollama_models()
        if ollama_models:
            default_chat_index = ollama_models.index("llama3:8b") if "llama3:8b" in ollama_models else 0
            model_name = st.selectbox("Select a Chat Model", ollama_models, index=default_chat_index)
            
            default_summary_index = ollama_models.index("llama3:8b") if "llama3:8b" in ollama_models else 0
            summary_model_name = st.selectbox("Select a Summary Model", ollama_models, index=default_summary_index)
            
            default_embed_index = ollama_models.index("mxbai-embed-large") if "mxbai-embed-large" in ollama_models else 0
            embeddings_model_name = st.selectbox("Select an Embeddings Model", ollama_models, index=default_embed_index)
        else:
            st.warning("Ollama not detected. Please enter model names manually.")
            model_name = st.text_input("Ollama Chat Model Name", "llama3")
            summary_model_name = st.text_input("Ollama Summary Model Name", "llama3")
            embeddings_model_name = st.text_input("Ollama Embeddings Model Name", "mxbai-embed-large")
        
        if model_name:
            st.session_state.model_config = {
                "model_name": model_name,
                "summary_model_name": summary_model_name,
                "embeddings_model_name": embeddings_model_name
            }
        
        st.header("3. Start Research")
        if st.button("Start Research Pipeline") and start_year <= end_year:
            if not st.session_state.model_config.get("model_name"):
                st.error("Please configure the AI model before starting.")
            else:
                st.session_state.research_done = False
                st.session_state.messages = []
                st.session_state.used_sources = set()
                checkpoint_file = f"checkpoint_{st.session_state.session_id}.json"
                if os.path.exists(checkpoint_file):
                    os.remove(checkpoint_file)
                with st.spinner("Running deep research pipeline..."):
                    domains = [d.strip() for d in domains_str.split("\n") if d.strip()]
                    initial_state = {
                        "query": query, 
                        "domains": domains, 
                        "num_references_per_domain": num_references,
                        "start_year": start_year,
                        "end_year": end_year
                    }
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
                        model_name=model_config["model_name"],
                        embeddings_model_name=model_config.get("embeddings_model_name")
                    )
                    retrieved_docs = retriever.get_relevant_documents(prompt)
                    
                    for doc in retrieved_docs:
                        if "url" in doc.metadata:
                            st.session_state.used_sources.add(doc.metadata["url"])
                        elif "sources" in doc.metadata:
                            for source_url in doc.metadata["sources"]:
                                st.session_state.used_sources.add(source_url)

                    context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
                    
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", "You are an AI research assistant. Answer based on the following context from academic papers:\n\n{context}\n\nIf the answer isn't in the context, say so."),
                        ("human", "{question}")
                    ])
                    chain = prompt_template | llm
                    response = chain.invoke({"context": context, "question": prompt})
                    response_content = response.content
                    st.markdown(response_content)
            
            st.session_state.messages.append({"role": "assistant", "content": response_content})
        
        with st.expander("Export Options"):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Export Chat & Used References"):
                    pdf_bytes = generate_pdf_report(
                        chat_history=st.session_state.messages, 
                        used_sources=sorted(list(st.session_state.used_sources))
                    )
                    st.download_button(label="Download Q&A PDF", data=pdf_bytes, file_name="chat_history.pdf", mime="application/pdf")
            with col2:
                if st.button("Export Full Bibliography"):
                    bib_pdf_bytes = generate_bibliography_pdf(
                        all_sources=st.session_state.final_state.get('direct_pdf_urls', [])
                    )
                    st.download_button(label="Download Bibliography PDF", data=bib_pdf_bytes, file_name="full_bibliography.pdf", mime="application/pdf")

            if st.button("Generate Pipeline Diagram"):
                final_state = st.session_state.final_state
                mermaid_code = generate_mermaid_diagram(
                    query=final_state['query'], 
                    domains=final_state['domains'],
                    num_refs=final_state['num_references_per_domain'], 
                    found_urls=len(final_state['direct_pdf_urls']),
                    start_year=final_state['start_year'],
                    end_year=final_state['end_year']
                )
                st.code(mermaid_code, language="mermaid")
    else:
        st.info("Configure your research and AI model in the sidebar, then click 'Start Research Pipeline'.")

if __name__ == "__main__":
    main()