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
import arxiv

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---

# Corrected Scopus to Elsevier, as it's the publisher.
DEFAULT_PUBLISHERS = ["IEEE", "ACM", "Springer", "Elsevier"]

# --- RAPTOR IMPLEMENTATION ---

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

    # --- START OF FIX 1 ---
    def _summarize_cluster(self, cluster_docs: List[Document]) -> Tuple[str, dict]:
        context = "\n\n---\n\n".join([doc.page_content for doc in cluster_docs])
        
        # Create a proper prompt template with a placeholder
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are an AI assistant that summarizes academic texts. Create a concise, abstractive summary of the following content, synthesizing the key information."),
            HumanMessage(content="Please summarize the following content:\n\n{context}")
        ])
        
        # Create a chain by piping the prompt and the model
        chain = prompt | self.llm
        
        # Invoke the chain with the context variable
        response = chain.invoke({"context": context})
        
        summary = response.content
        aggregated_sources = list(set(doc.metadata.get("url", "Unknown Source") for doc in cluster_docs))
        combined_metadata = {"sources": aggregated_sources}
        return summary, combined_metadata
    # --- END OF FIX 1 ---
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        return self.vector_store.similarity_search(query, k=k) if self.vector_store else []
    
    def as_retriever(self) -> BaseRetriever:
        return RAPTORRetriever(raptor_index=self)

# --- STATE DEFINITION FOR LANGGRAPH (No changes here) ---
class ResearchState(TypedDict):
    query: str
    publishers: List[str]
    num_references: int
    start_year: int
    end_year: int
    papers_by_publisher: Dict[str, List[str]]
    path_to_metadata_map: Dict[str, Dict[str, str]]
    extracted_docs: list
    raptor_index: Any
    conversation_history: Annotated[List[BaseMessage], operator.add]
    generation: str

# --- LANGGRAPH NODES AND GRAPH DEFINITION ---
def start_search_node(state: ResearchState) -> ResearchState:
    st.write("Starting research process...")
    return {"papers_by_publisher": {}, "path_to_metadata_map": {}, "extracted_docs": []}

def arxiv_search_and_filter_node(state: ResearchState) -> ResearchState:
    st.write("Stage 1: Searching arXiv and filtering by publisher...")
    
    # --- START OF FIX 2 ---
    # Corrected filter logic for Elsevier (owner of Scopus)
    filter_criteria = {
        "IEEE": lambda r: (r.journal_ref and "IEEE" in r.journal_ref) or (r.doi and "10.1109" in r.doi),
        "IEEE Explorer": lambda r: (r.journal_ref and "IEEE Explorer" in r.journal_ref)   ,      
        "IEEE Transactions": lambda r: (r.journal_ref and "IEEE Transactions" in r.journal_ref)   ,      
        "ACM": lambda r: (r.journal_ref and "ACM" in r.journal_ref) or (r.doi and "10.1145" in r.doi),
        "Springer": lambda r: (r.journal_ref and "Springer" in r.journal_ref) or (r.doi and "10.1007" in r.doi),
        "Elsevier": lambda r: (r.journal_ref and "Elsevier" in r.journal_ref) or (r.doi and "10.1016" in r.doi)
    }
    # --- END OF FIX 2 ---
    
    selected_publishers = state["publishers"]
    if not selected_publishers:
        st.warning("No publishers selected. Please select at least one.")
        return {"papers_by_publisher": {}}
    
    query_terms = state["query"]
    start_dt = datetime(state['start_year'], 1, 1)
    end_dt = datetime(state['end_year'], 12, 31)
    
    st.write(f"Searching for: '{query_terms}'...")
    search = arxiv.Search(
        query=query_terms,
        max_results=state["num_references"] * 15,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    found_papers = {pub: [] for pub in selected_publishers}
    total_found = 0
    total_needed = state["num_references"]
    
    results_iterator = search.results()
    search_progress = st.progress(0, text="Filtering arXiv results...")
    
    st.write("Iterating through results to find matches...")
    
    try:
        for i, result in enumerate(results_iterator):
            if total_found >= total_needed:
                st.write(f"Reached the desired number of {total_needed} references.")
                break

            if i % 20 == 0:
                progress_val = total_found / total_needed if total_needed > 0 else 0
                search_progress.progress(progress_val, text=f"Scanned {i+1} papers | Found {total_found}/{total_needed} matches")

            if not (start_dt.date() <= result.published.date() <= end_dt.date()):
                continue
                
            for pub in selected_publishers:
                if filter_criteria.get(pub)(result):
                    if result.pdf_url and result.pdf_url not in [url for urllist in found_papers.values() for url in urllist]:
                        found_papers[pub].append(result.pdf_url)
                        total_found += 1
                        break
    except arxiv.UnexpectedEmptyPageError:
        st.info("Processed all available results from arXiv for this query.")

    search_progress.progress(1.0, text="Filtering complete.")
    
    st.write("--- Search Results by Publisher ---")
    for pub, papers in found_papers.items():
        st.write(f"- {pub}: Found {len(papers)} papers.")
        
    if total_found == 0:
        st.warning("Could not find any papers on arXiv matching your criteria. Try broadening your search.")
        
    return {"papers_by_publisher": found_papers}

def download_pdfs_node(state: ResearchState) -> ResearchState:
    st.write("Stage 2: Downloading valid PDFs...")
    path_to_metadata_map = {}
    if not os.path.exists("temp_pdfs"):
        os.makedirs("temp_pdfs")
    
    papers_by_publisher = state["papers_by_publisher"]
    all_urls = [(url, pub) for pub, urls in papers_by_publisher.items() for url in urls]
    
    total_urls = len(all_urls)
    if total_urls == 0:
        st.warning("No PDF URLs found to download.")
        return {"path_to_metadata_map": {}}
        
    download_progress = st.progress(0, text="Starting download...")

    for i, (url, publisher) in enumerate(all_urls):
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
                path_to_metadata_map[filename] = {"url": url, "publisher": publisher}
            else:
                st.warning(f"Skipped non-PDF file at {url} (Content-Type: {content_type})")
        except requests.exceptions.RequestException as e:
            st.warning(f"Failed to download {url}: {e}")
            
    return {"path_to_metadata_map": path_to_metadata_map}

def extract_text_node(state: ResearchState) -> ResearchState:
    st.write("Stage 3: Extracting text from PDFs...")
    all_docs = []
    path_to_metadata_map = state["path_to_metadata_map"]
    for path, metadata in path_to_metadata_map.items():
        try:
            loader = PyPDFLoader(path)
            pages = loader.load_and_split()
            for page in pages:
                page.metadata["url"] = metadata["url"]
                page.metadata["publisher"] = metadata["publisher"]
            all_docs.extend(pages)
        except Exception as e:
            st.warning(f"Could not read or parse PDF from {metadata['url']}. It may be corrupted. Error: {e}")
            
    st.write(f"Successfully extracted {len(all_docs)} document chunks.")
    return {"extracted_docs": all_docs}

def get_llm_and_embeddings(model_name: str, embeddings_model_name: Optional[str] = None):
    llm = ChatOllama(model=model_name, temperature=0.3)
    embed_model = embeddings_model_name if embeddings_model_name else model_name
    embeddings = OllamaEmbeddings(model=embed_model)
    return llm, embeddings

def build_raptor_index_node(state: ResearchState) -> ResearchState:
    st.write("Stage 4: Building RAPTOR index... This may take some time.")
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

# --- GRAPH DEFINITION (No changes here) ---
builder = StateGraph(ResearchState)
builder.add_node("start_search", start_search_node)
builder.add_node("arxiv_search_and_filter", arxiv_search_and_filter_node)
builder.add_node("download_pdfs", download_pdfs_node)
builder.add_node("extract_text", extract_text_node)
builder.add_node("build_raptor_index", build_raptor_index_node)

builder.add_edge(START, "start_search")
builder.add_edge("start_search", "arxiv_search_and_filter")
builder.add_edge("arxiv_search_and_filter", "download_pdfs")
builder.add_edge("download_pdfs", "extract_text")
builder.add_edge("extract_text", "build_raptor_index")
builder.add_edge("build_raptor_index", END)
graph = builder.compile()

# --- HELPER FUNCTIONS & UI (No changes here) ---
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

def generate_bibliography_pdf(papers_by_publisher: Dict[str, List[str]]) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', size=16)
    pdf.cell(0, 10, txt="Full Bibliography of Scraped Articles", ln=True, align='C')
    pdf.ln(10)
    
    for publisher, urls in papers_by_publisher.items():
        if urls:
            pdf.set_font("Arial", 'B', size=14)
            pdf.cell(0, 10, txt=f"--- {publisher} ---", ln=True, align='L')
            pdf.ln(5)
            pdf.set_font("Arial", size=10)
            for i, url in enumerate(urls):
                pdf.multi_cell(0, 8, f"{i+1}. {url}")
            pdf.ln(5)
            
    return pdf.output(dest='S').encode('latin-1')

def generate_mermaid_diagram(query: str, publishers: List[str], num_refs: int, papers_by_publisher: Dict[str, List[str]], start_year: int, end_year: int) -> str:
    diagram = f"""graph TD;
    A[Start: User Input] --> B(LangGraph Pipeline);
    B --> C[Search arXiv & Filter];
    """
    
    result_nodes = []
    for pub in publishers:
        count = len(papers_by_publisher.get(pub, []))
        if count > 0:
            node_id = f"R_{pub.replace(' ', '')}"
            result_nodes.append(node_id)
            diagram += f'    C --> {node_id}["Found {count} {pub} Papers"];\n'
    
    if result_nodes:
         diagram += f"    {{{', '.join(result_nodes)}}} --> D[Download & Validate PDFs];\n"
    else:
        diagram += "    C --> D[Download & Validate PDFs];\n"

    diagram += f"""    D --> E[Extract Text];
    E --> F[Build RAPTOR Index];
    F --> G[Conversational QA];
    subgraph Parameters;
        P1("Query: {query}");
        P2("Publishers: {', '.join(publishers)}");
        P3("Target References: {num_refs}");
        P4("Years: {start_year}-{end_year}");
    end;
    """
    return diagram

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
    st.markdown("Powered by Ollama 🦙 and arXiv 📄")

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.research_done = False
        st.session_state.final_state = None
        st.session_state.model_config = {}
        st.session_state.used_sources = set()

    with st.sidebar:
        st.header("1. Research Parameters")
        query = st.text_input("Academic Topic", "indoor air quality monitoring using machine learning")
        
        # Updated publisher list in the UI
        publishers = st.multiselect(
            "Filter by Publisher (via arXiv metadata)",
            options=["IEEE", "ACM", "Springer", "Elsevier"],
            default=DEFAULT_PUBLISHERS
        )
        st.info("This filters arXiv papers that have a journal reference or DOI matching the selected publishers. 'Elsevier' is used as a proxy for papers indexed in Scopus.")

        num_references = st.slider("Total Desired References", 1, 100, 10)
        
        current_year = datetime.now().year
        start_year = st.number_input("Start Year", min_value=1991, max_value=current_year, value=2020)
        end_year = st.number_input("End Year", min_value=1991, max_value=current_year, value=current_year)
        
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
                    initial_state = {
                        "query": query, 
                        "publishers": publishers, 
                        "num_references": num_references,
                        "start_year": start_year,
                        "end_year": end_year,
                        "conversation_history": []
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
                        papers_by_publisher=st.session_state.final_state.get('papers_by_publisher', {})
                    )
                    st.download_button(label="Download Bibliography PDF", data=bib_pdf_bytes, file_name="full_bibliography.pdf", mime="application/pdf")

            if st.button("Generate Pipeline Diagram"):
                final_state = st.session_state.final_state
                mermaid_code = generate_mermaid_diagram(
                    query=final_state['query'], 
                    publishers=final_state['publishers'],
                    num_refs=final_state['num_references'], 
                    papers_by_publisher=final_state['papers_by_publisher'],
                    start_year=final_state['start_year'],
                    end_year=final_state['end_year']
                )
                st.code(mermaid_code, language="mermaid")
    else:
        st.info("Configure your research and AI model in the sidebar, then click 'Start Research Pipeline'.")

if __name__ == "__main__":
    main()