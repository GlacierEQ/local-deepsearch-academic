# ![deep-research](https://github.com/user-attachments/assets/b98d01cb-0d7d-4cd9-a117-8af29d0b47e8)

## Academic Deep Search & QA 📚🧠

I built this tool because I wanted a way to quickly dive into a new research topic, pull down the relevant papers, and start asking questions right away—without spending days on manual searching and reading.

This app is a personal research assistant that automates the grunt work. You give it a topic, and it builds a custom, chat-ready knowledge base from real academic articles, all powered by local LLMs through Ollama.

## So, What Does It Actually Do? 🤔

*   🔍 **Targeted Scraping:** You provide a topic and academic domains (like `arxiv.org`). The app uses specialized APIs (like the Arxiv API) where possible to find actual article pages. For other sites, it intelligently scrapes the landing pages to find the direct PDF links, avoiding irrelevant blog posts and HTML pages.
*   📥 **Robust Downloading & Validation:** It attempts to download only legitimate PDFs by checking the file headers first. This prevents the common issue of accidentally saving HTML login pages as PDFs.
*   📄 **Text Extraction:** It cracks open the PDFs it successfully downloads and pulls out all the text content, getting it ready for the AI.
*   🦖 **Advanced Indexing with RAPTOR:** This is the cool part. Instead of just chunking the text, it uses the **RAPTOR** method to create a multi-level tree of summaries. This means it understands the papers from tiny details all the way up to high-level concepts, leading to much better answers.
*   💬 **Conversational QA:** Once the index is built, you get a chat interface (thanks to Streamlit!) where you can ask complex questions and get answers synthesized from the papers it just read.
*   📤 **Export Your Findings:** You can export your Q&A session with a bibliography of *cited sources*, a separate document with the *full bibliography* of every paper found, or a Mermaid diagram that visualizes the pipeline run.

## The Tech Stack 🛠️

*   **Backend Logic:** [LangGraph](https://langchain-ai.github.io/langgraph/) 🦜🔗 for creating a resilient, stateful pipeline.
*   **Indexing:** A custom **RAPTOR** implementation for intelligent, multi-level retrieval.
*   **AI Models:** Plugs into your local models via [Ollama](https://ollama.ai/). You can configure separate models for chat, summarization, and embeddings.
*   **Scraping:** [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) and specialized wrappers like `ArxivAPIWrapper`.
*   **Frontend:** [Streamlit](https://streamlit.io/) 🎈 for a fast, interactive web UI.
*   **The Glue:** The whole thing is tied together with [LangChain](https://www.langchain.com/).

## Getting Started 🚀

Ready to give it a spin? Here’s how to get it running on your machine.

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/academic-deep-search.git
cd academic-deep-search
```

2. Set Up Your Environment
This project uses a requirements.txt file, so setting up is a breeze. I recommend using a virtual environment.
code

# Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
# Install all the goodies
pip install -r requirements.txt
```

3. Get Ollama Running
This app runs entirely on local models via Ollama.
Install Ollama.
Pull a few models for the different tasks. For the best experience, I recommend a good chat model and a specialized embedding model:
code
Bash
# A great, fast model for chat and summarization
ollama pull qwen3:1.7b

# A top-tier model specifically for embeddings
ollama pull mxbai-embed-large
Make sure the Ollama server is running in the background.
4. Fire It Up!
Run the Streamlit app from your terminal:

```bash
streamlit run app.py
```
Your browser should open with the app ready to go! Configure your topic and select your desired Ollama models in the sidebar, then kick off the research.

Use Cases & Who This Is For

- I built this with a few people in mind:
Students & Academics: Need to write a literature review? Point this at a topic and get a massive head start. Quickly find key themes and ask targeted questions to build your arguments.
- Data Scientists & Engineers: Exploring a new machine learning architecture or a novel algorithm? Let the app grab the foundational papers and get you up to speed in minutes, not hours.
- Curious Minds: Just want to learn about something cool like quantum computing or cellular biology? This is a fun, interactive way to dive deep into a topic.

Got an idea? Found a bug? Feel free to open an issue or submit a pull request! I'd love to see what the community can build on top of this. Let's make research less of a chore.