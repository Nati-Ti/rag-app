# Customer Support ChatBot with RAG

This app implements a Customer Support ChatBot using Retrieval-Augmented Generation (RAG) techniques, focusing on Sentence Window Retrieval and Auto-Merging Retrieval methods. The chatbot retrieves and processes relevant information to generate accurate and contextually relevant responses.

## Basic RAG Implementation

- **`embed_pinecone.py`**
  - **Function:** Chunks data from a JSON file, embeds the chunks, and stores them in the Pinecone vector database.
  - **Tasks:**
    - Chunk data from the JSON file
    - Embed the chunked data
    - Store embedded data in Pinecone

- **`app_rag.py`**
  - **Function:** Handles user queries, fetches relevant data from Pinecone, feeds the context to the language model, and returns the generated output.
  - **Tasks:**
    - Accept user queries
    - Fetch data from Pinecone based on the query
    - Provide context to the language model
    - Return generated responses

## Sentence Window Retrieval

- **`swr_embed_pinecone.py`**
  - **Function:** Chunks data by sentence, creates sentence windows, embeds these windows, and stores them in Pinecone.
  - **Tasks:**
    - Chunk data by sentence
    - Create sentence windows from the chunks
    - Embed the sentence windows
    - Store embeddings in Pinecone

- **`app_swr.py`**
  - **Function:** Handles user queries, fetches data from Pinecone using sentence windows, performs similarity checks, feeds context to the language model, and returns the generated response.
  - **Tasks:**
    - Accept user queries
    - Fetch sentence windows from Pinecone
    - Compute similarity with sentence windows
    - Provide context to the language model
    - Return generated responses

## Auto-Merging Retrieval

- **`app_amr.py`**
  - **Function:** Manages context data files, creates hierarchical data chunks, stores and indexes these chunks, performs similarity checks, and provides relevant context to the language model.
  - **Tasks:**
    - Load context data file
    - Create hierarchical data chunks with 3 levels of sizes `[512, 256, 128]`
    - Prepare storage for context nodes
    - Store leaf node indices in a vector store
    - Perform similarity checks with leaf nodes
    - Return top similar leaf nodes and, if above threshold, the parent node
    - Feed the context to the language model
    - Return the generated response

## Evaluation

- **`eval.py`**
  - **Function:** Tests the system with sample data, evaluates performance using metrics such as F1-Score, BLEU, and ROGUE, and compares results from `app_rag.py` and `app_swr.py`.
  - **Tasks:**
    - Define test data (questions and ground truth)
    - Perform F1-Score, BLEU, and ROGUE evaluations
    - Compare results from `app_rag.py` and `app_swr.py`
    - Return evaluation results
