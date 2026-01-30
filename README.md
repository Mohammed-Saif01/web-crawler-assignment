This assignment is an AI-powered chatbot that scrapes content from a user-provided website URL, processes the text into embeddings, and allows users to ask questions based on that content. It uses a RAG (Retrieval-Augmented Generation) pipeline to ensure accurate, context-aware answers.

Architecture
1. **Ingestion:** The app uses `WebBaseLoader` to crawl the target URL.
2. **Chunking:** Text is split into 1000-character chunks with 200-character.
3. **Embedding:** Chunks are converted to vectors.
4. **Storage:** Vectors are stored locally using **FAISS** for fast similarity search.
5. **Retrieval & Generation:** Relevant chunks are retrieved and passed to **Gemini 1.5 Flash** to generate the final answer.

Frameworks used
1. LangChain
2. langchain_community
3. langchain_google_genai
4. langchain_core

I have used Gemini 1.5 Flash LLM for its high speed, large context window, and cost-effectiveness (free tier) for this assignment.
I have used Vector DB (FAISS) because it is lightweight, runs locally without external server setup, and supports saving indices to disk (persistence).
I have Embeddings (Google Generative AI to maintain a unified Google ecosystem. It offers optimized semantic search capabilities compatible with the Gemini LLM and supports multilingual retrieval better than standard open-source models.

**Setup and run instructions**
1. **Clone the repository:**
   git clone https://github.com/Mohammed-Saif01/web-crawler-assignment.git
   
3. **Install dependencies**
 pip install -r requirements.txt

3.**Configure API Keys: Create a .env file in the root directory and add**
   GOOGLE_API_KEY=your_api_key_here
   
4.**Run the application:**
   streamlit run app.py

limitation :
1. if the website uses heavy JavaScript (CSR) to load content WebBaseLoader might miss some sections.
2. the application uses local memory for the vector store, so restarting the app clears the indexed website (unless persistence logic is explicitly triggered).
3. since my GOOGLE AI STUDIO's billing setup is under verification so the results might not be visible immediatly, it will take some time to get complete(probably 48 hours).
