# PDF Chatbot with Citations

A Retrieval-Augmented Generation (RAG) application that ingests PDF documents, retrieves relevant chunks based on semantic and keyword search, and generates answers with precise citations back to the source document.

## Overview

The system is designed to handle complex PDF layouts including tables, figures, and multi-level headings. It preserves document structure during parsing, allowing the final generated answers to cite specific sections and page numbers.

The pipeline consists of three stages:

1. **Ingestion**: Parse the PDF, extract structural elements, chunk the content, generate embeddings, and store them in a vector database.
2. **Retrieval**: Convert a user query into a vector, perform similarity search against the stored chunks, and return the most relevant results.
3. **Generation**: Pass the retrieved chunks and the original query to a large language model, which produces a structured answer with inline citations.

## Project Structure

```
PDF-Chatbot-with-Citations/
├── docs/                       # Source PDF files
├── ingestion/
│   ├── loader.py               # Parses PDFs into structured elements
│   ├── chunk_processing.py     # Splits elements into chunks with metadata
│   ├── tab_img_processor.py    # Handles tables and images during parsing
│   └── embed.py                # Generates embeddings and upserts to vector store
├── retrieval/
│   └── retrieve.py             # Query handling and similarity search
├── main.py                     # Entry point for the CLI application
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (API keys)
└── .gitignore
```

## Requirements

- Python 3.12 or higher
- A Pinecone account and API key
- An OpenAI-compatible API key for embeddings and generation
- PDFs placed in the `docs/` directory

## Setup

1. Clone the repository and navigate into the project directory.

2. Create and activate a virtual environment.

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies.

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add the required keys.

```
PINECONE_API_KEY=your_pinecone_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
```

5. Add your PDF files to the `docs/` directory.

## Usage

### Step 1: Ingest Documents

Run the ingestion script to parse the PDFs, generate embeddings, and populate the vector index. This only needs to be run once per document, or again if the documents change.

```bash
python -m ingestion.embed
```

### Step 2: Run the Chatbot

Start the CLI application.

```bash
python main.py
```

The application will prompt for a query. Type your question and press Enter. The system will retrieve relevant chunks and generate an answer with citations. Type `exit` or `quit` to stop the application.

## How It Works

### Ingestion Pipeline

The ingestion process begins with `UnstructuredLoader`, which parses the PDF using a high-resolution strategy to preserve layout information. Each element (paragraph, title, table, figure caption) is returned as a separate Document with metadata including page number, category, and heading information.

A heading tracking pass walks through the elements in order and attaches the current heading and subheading to each element's metadata. This ensures that every subsequent chunk inherits the section context it belongs to.

The documents are then split using `RecursiveCharacterTextSplitter`, which breaks long sections into smaller overlapping chunks while preserving the heading metadata on each chunk.

Finally, each chunk is embedded and upserted into a Pinecone index. Metadata is sanitized before upsert to comply with Pinecone's size and type constraints.

### Retrieval

When a query is received, it is converted into a vector using the same embedding model. A similarity search is performed against the Pinecone index, returning the top-k most relevant chunks along with their metadata.

### Generation

The retrieved chunks are formatted into a structured context and passed to the LLM along with a system prompt. The prompt instructs the model to:

- Answer strictly from the provided context
- Cite every factual claim with the format `[filename, heading, sub-heading, page-number]`
- Refuse to answer when the context is insufficient
- Return the answer as structured JSON for easy parsing

## Configuration

The following parameters can be adjusted in the ingestion and retrieval scripts:

- `chunk_size`: Target size of each chunk in characters (default: 600)
- `chunk_overlap`: Overlap between consecutive chunks (default: 0)
- `top_k`: Number of chunks retrieved per query (default: 5)
- `embedding_model`: Model used for vectorization
- `llm_model`: Model used for answer generation

## Metadata Fields

Each chunk stored in the vector database contains the following metadata fields:

| Field | Description |
|-------|-------------|
| filename | Name of the source PDF |
| page_number | Page number in the original document |
| heading | Main section heading |
| sub-heading | Subsection heading |
| category | Element type (NarrativeText, Table, etc.) |

## Limitations

- Pinecone has a 40 KB metadata limit per vector. Very large chunks must be trimmed before upsert.
- Tables are extracted as text and may lose some formatting.
- The quality of heading detection depends on the structure of the source PDF.
- The system is designed for PDFs with a clear text layer. Scanned documents require OCR.

## License

This project is intended for educational and research purposes.
