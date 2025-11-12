import os
import re
from typing import List, Dict

# Assumed libraries based on usage
from langdetect import detect # Used in detect_language_safe
import pyarabic.araby as araby # Used in normalize_arabic
import arabic_reshaper # Used in rag_pipeline for display
from bidi.algorithm import get_display # Used in rag_pipeline for display

# LangChain and related imports (estimated)
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    UnstructuredPDFLoader, # Used instead of PyMuPDFLoader/PDFMinerLoader
)
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker # Assumed path for SemanticChunker
from langchain_community.vectorstores.pgvector import DistanceStrategy # For PGVector distance_strategy

def connect_to_db ():
    host = ''
    port = ''
    username = ''
    password = ''
    database_schema = ''

    connection_string = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database_schema}"
    # print("done")

    # Connect to PostgreSQL database
    conn = psycopg2.connect(database=database_schema, user=username, host=host, port=port, password=password)
    cur = conn.cursor()
    print("Connected to Vector DB")
    return connection_string, cur, conn


def load_embedding_model():
    # Instantiate the Embedding Model
    model_name = "BAAI/bge-m3"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceBgeEmbeddings(model_name=model_name,
                                          model_kwargs=model_kwargs,
                                          encode_kwargs=encode_kwargs,
                                          query_instruction="Represent this sentence for searching relevant passages:"
                                          )
    return embeddings

def load_file(file_path: str) -> List[Document]:
    """
    Loads a document based on file type (PDF, Word, Excel, CSV).

    Args:
        file_path (str): Path to the input file.

    Returns:
        List[Document]: The list of Document objects.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.split(file_path)[1].lower()
    source_name = os.path.basename(file_path) # Extracts file name for metadata

    # Initialize the loader
    if ext == ".pdf":
        print("Detected file type: PDF")
        # loading PDF using PyMuPDFLoader good for multilingual tasks
        # loader = PyMuPDFLoader(file_path)
        
        # loading PDF using PDFMinerLoader good tables, not good enough for multiling
        # loader = PDFMinerLoader(file_path)

        # loading PDF using UnstructuredPDFLoader good for OCR and tab
        loader = UnstructuredPDFLoader(
            file_path,
            strategy="ocr_only",
            extract_images=True,
            extract_table=True,
            ocr_languages="ara+eng",
        )
    elif ext == ".csv":
        print("Detected file type: CSV")
        loader = CSVLoader(file_path=file_path)
    elif ext in [".docx", ".doc"]:
        print("Detected file type: Word Document")
        loader = UnstructuredWordDocumentLoader(file_path)
    elif ext in [".xls", ".xlsx"]:
        print("Detected file type: Excel Spreadsheet")
        loader = UnstructuredExcelLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    docs = loader.load()

    # Add document type to metadata
    for doc in docs:
        doc.metadata["source"] = source_name
        doc.metadata["document_type"] = ext.lstrip('.') # .PDF -> "pdf"

    return docs


def prepare_documents_for_rag(unstructured_docs: List[Document], source_name: str) -> List[Dict]:
    """
    Cleans, normalizes, and detects language in documents for RAG preparation.
    Returns a list of {"text": ..., "metadata": ...} dicts.
    """
    def clean_text(text: str) -> str:
        """Basic whitespace and artifact cleanup."""
        if not text:
            return ""
        text = text.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        patterns = [
            r"page \d+ of \d+", r"confidential", r"https?://[^\s]+",
            r"\bcompany name\b", r"\bprivate & confidential\b"
        ]
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        return text

    def normalize_arabic(text: str) -> str:
        """Normalize Arabic characters and remove diacritics."""
    # --- Visible/Implied Helper Functions ---
        # text = de_diac_ar(text) # Removes tashkeel (diacritics)
        # text = normalize_teh_marbuta_ar(text)
        # text = normalize_alef_ar(text)
        # text = normalize_alef_maksura_ar(text)
        # ----------------------------------------
        
        # General Arabic normalization/cleanup as implemented in image 163337
        #text = re.sub(r'[\u0626]', '', text)
        text = re.sub(r'[أإآ]', 'ا', text)
        text = re.sub(r'[ة]', 'ه', text)
        text = re.sub(r'[يى]', 'ي', text)
        text = araby.strip_tashkeel(text)       
        return text

    def detect_language_safe(text: str) -> str:
        """Safely detect language."""
        try:
            return detect(text)
        except:
            return "unknown"

    def normalize_text(text: str, lang: str) -> str:
        """Apply language-specific normalization."""
        text = clean_text(text)
        if lang == "ar":
            text = normalize_arabic(text)
        return text

    cleaned_texts = []

    for doc in unstructured_docs:
        raw_text = getattr(doc, "page_content", "")
        if not raw_text or not isinstance(raw_text, str):
            continue

        lang = detect_language_safe(raw_text)
        normalized = normalize_text(raw_text, lang)

        if normalized:
            cleaned_texts.append({
                "text": normalized,
                "metadata": {
                    "source": source_name,
                    "language": lang,
                    **doc.metadata
                }
            })

    return cleaned_texts


def chunk_documents(cleaned_texts: List[Dict], chunk_size: int = 800, chunk_overlap: int = 150) -> List[Document]:
    """
    Splits cleaned text into semantic chunks and returns LangChain Document objects.
    Applies different chunking strategies depending on document type (Word, Excel, PDF, etc.).
    """
    final_docs: List[Document] = []
    for item in cleaned_texts:
        text = item["text"]
        doc_type = item["metadata"]["document_type"]

        # Define chunk size and overlap based on document type
        if doc_type == "pdf" or doc_type == "docx":
            chunk_size = 800 # Longer chunks for documents with more text
        elif doc_type in ["csv", "xlsx"]:
            chunk_size = 500 # Shorter chunks for tables to avoid mixing content
            chunk_overlap = 50
        else:
            chunk_size = 800
            chunk_overlap = 150

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "؟", ",", " ","!"] # Added separators from observation
        )
        chunks = splitter.split_text(text)

        for chunk in chunks:
            final_docs.append(Document(page_content=chunk, metadata=item["metadata"]))

    print(f"Chunking complete. Generated {len(final_docs)} chunks.")
    return final_docs


def chunk_documents_semantic(cleaned_texts: List[Dict], embeddings, threshold: int = 90) -> List[Document]:
    
    final_docs : list[Document] = []
    for item in cleaned_texts:
        content = item["text"]
        metadata = item["metadata"]

        # Create SemanticChunker per document
        text_splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type='percentile',
            breakpoint_threshold_amount=threshold
        )

        # Apply semantic chunking
        chunks = text_splitter.create_documents([content])

        for chunk in chunks:
            # Preserve original metadata
            chunk.metadata.update(metadata)
            final_docs.append(chunk)

    print(f"Semantic chunking complete. Generated {len(final_docs)} chunks.")
    return final_docs

def save_embeddings(final_docs, embeddings, connection_string):
    # Create a PGVector instance to house the documents and embeddings
    db_vectorization = PGVector.from_documents(
        documents = final_docs, # The documents we loaded from the Pand...
        embedding = embeddings, # Our instance of the embeddings class, ...
        collection_name = "", # The name of the table we want c...
        distance_strategy = DistanceStrategy.COSINE, # The distance stra...
        connection_string = connection_string) # The connection str...
    print("SAVED IN DATABASE")

def hybrid_retriever(embeddings, connection_string):
    store = PGVector(
        connection_string=connection_string,
        embedding_function=embeddings,
        collection_name="",
        distance_strategy=DistanceStrategy.COSINE
    )

    print("Connected to PGVector")

    dense = store.as_retriever(search_type="mmr", search_kwargs={"k": 20})
    sparse = store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 5, "score_threshold": 0.6})

    retriever = EnsembleRetriever(retrievers=[dense, sparse], weights=[0.7, 0.3])

    #query = "What seasonings are used to flavor the chicken breasts?" # Example query
    #docs = retriever.invoke(query)
    
    return retriever

def load_llm():
    ollama = Ollama(
        base_url='http://ollama:11434',
        model='gemma:latest',
        temperature=0.2
    )

    print("LOADED OLLAMA")
    return ollama

def rag_pipeline(ollama, retriever, query):
    # To show ARABIC in the terminal
    def contains_arabic(text: str) -> bool:
        return bool(re.search(r'[\u0600-\u06FF\u0750-\u077F]', text))

    def for_display(text: str) -> str:
        if contains_arabic(text):
            # Assumes 'get_display' and 'arabic_reshaper' are imported/defined elsewhere
            return get_display(arabic_reshaper.reshape(text))
        return text # No reshaping for English

    # Prompt Template
    prompt_RAG = """
        You are a helpful and accurate assistant. You receive content retrieved from user-provided documents and must answer the user's question according to the following rules:
        
        1. **Use only the retrieved content**: Do not make assumptions or add external information not present in the retrieved text.
        2. **Handle poorly formatted or dense text carefully**: The content may include messy formatting, tables, or inline data. Parse it and summarize or reformat to make it readable.
        3. **Extract values embedded in text**: If numbers, dates, names, or other important information are embedded within sentences or paragraphs, extract them clearly.
        4. **Multilingual support**: The retrieved content may be in English or Arabic. Detect the language and respond in the same language as the question, if possible, or the language of the majority of the retrieved text.
        5. **Answer clearly and concisely**: Summarize or reformat dense content for readability if necessary, but ensure the information is not lost.
        6. **Preserve data integrity**: Do not alter numbers, dates, names, or other factual values. Report them exactly as they appear in the source.
        7. **Explicitly cite sources if available**: If the metadata or source name is provided, include it in your response to help the user trace the answer back to the original document.
        
        Context:
        {context}
        
        Question:
        {question}
        """

    prompt_RAG_tempate = PromptTemplate(
        template=prompt_RAG, input_variables=["question", "context"]
    )

    chain_type_kwargs = {"prompt": prompt_RAG_tempate}

    # Assumes RetrievalQA is imported from Langchain or similar library
    qa_chain = RetrievalQA.from_chain_type(
        llm=ollama,
        chain_type="stuff",
        chain_type_kwargs=chain_type_kwargs,
        retriever=retriever,
        return_source_documents=True
    )

    responses = qa_chain.invoke({"query": query}, callbacks=None)

    source_documents = responses["source_documents"]
    # print(source_documents)
    source_content = [doc.page_content for doc in source_documents]
    source_metadata = [doc.metadata for doc in source_documents]

    # Construct a single string with the LLM output and the source titles and urls
    def construct_result_with_sources():
        result = responses["result"]
        result += "\n\n"
        result += "Sources used:"

        for i in range(len(source_content)):
            result += "\n\n"
            # result += source_metadata[i]["document_type"]
            # result += "\n\n"
            # result += source_metadata[i]["uri"] # Assuming 'uri' key exists in metadata
            # result += f'Source {i+1}: {source_metadata[i].get("source", "Unknown")} (Type: {source_metadata[i].get("document_type", "Unknown")})' # More robust access
            
        return result

    response = construct_result_with_sources()
    print(for_display(response)) # Uses the Arabic display function


    return response

if __name__ == "__main__":
    while True:
        print("\n RAG Chatbot Options")
        print("1. Chat with existing documents")
        print("2. Add and chat with a new document")
        print("0. Exit")

        choice = input("Enter your choice (1/2/0): ").strip()

        if choice == "1":
            # Chat with existing documents
            connection_string, cur, conn = connect_db()
            embeddings = embedding_model()
            retriever = hybrid_retriever(embeddings, connection_string)
            ollama = load_llm()


            while True:
                query = input("\n Enter your query: ").strip()
                if not query:
                    print(" Empty query. Try again.")
                    continue

                    rag_pipline(ollama, retriever, query)

                    again = input("\n Do you want to ask another question? (y/n): ").strip().lower()
                    if again != "y":
                        break

        elif choice == "2":
            # Add and embed new document
            file_path = input("\n Enter path to your document (PDF, Word, Excel, CSV): ").strip()

            if not os.path.exists(file_path):
                print(f"\n File not found. Please check the path.")
                continue

            try:
                docs = load_file_path(file_path)
                print(f"\n Loaded {len(docs)} pages from: {os.path.basename(file_path)}")

                # Step 1: Clean & normalize
                cleaned_texts = prepare_documents_for_rag(docs, source_name=os.path.basename(file_path))

                # Step 2: chunk
                final_docs = chunk_documents(cleaned_texts)

                # Step 3: Embed & store
                connection_string, cur, conn = connect_db()
                embeddings = embedding_model()
                save_embeddings(final_docs, embeddings, connection_string)

                retriever = hybrid_retriever(embeddings, connection_string)
                ollama = load_llm()
            
                while True:
                    query = input("\n Enter your query: ").strip()
                    if not query:
                        print(f"\n Empty query. Try again.")
                        continue
                
                    try:
                        rag_pipline(ollama, retriever, query)
                
                        again = input("\n Do you want to ask another question? (y/n): ").strip().lower()
                        if again != "y":
                            break
                
                    except Exception as e:
                        print(f" Error: {e}")
                
                elif choice == "0":
                    print(f"\n Exiting. Goodbye!")
                    break
                
                else:
                    print(f"\n Invalid choice. Please enter 1, 2, or 0.")


