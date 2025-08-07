import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config.settings import GOOGLE_API_KEY


def build_qa_chain(file_path: str, index_path="vectorstore/index"):
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set in config/settings.py")
    
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

    # Determine loader based on file extension
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in [".docx", ".doc"]:
        loader = UnstructuredWordDocumentLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    # Load documents
    documents = loader.load()

    # Check if FAISS index exists
    if os.path.exists(index_path):
        print(" Loading existing FAISS index...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        print("[+] Creating new FAISS index...")
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(documents)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_documents(chunks, embeddings)

        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        vectorstore.save_local(index_path)
        print(f" Saved FAISS index to {index_path}")

    retriever = vectorstore.as_retriever()
    llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )

    return qa_chain
