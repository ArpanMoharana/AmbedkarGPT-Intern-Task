
import os
import sys

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA


# ---------- Config ----------
SPEECH_FILE = "speech.txt"
CHROMA_PERSIST_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
OLLAMA_MODEL = "mistral"  # ensure you ran `ollama pull mistral`
RETRIEVAL_K = 4

# ---------- Functions ----------
def create_or_load_vectorstore(speech_path: str, persist_dir: str):     # Load speech text, split into chunks, create embeddings, and persist or reuse a Chroma vectorstore

    embed = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if os.path.exists(persist_dir) and any(os.scandir(persist_dir)):
        print(f"Loading existing Chroma DB from '{persist_dir}'...")
        vectordb = Chroma(persist_directory=persist_dir, embedding_function=embed)
        return vectordb


    if not os.path.exists(speech_path):
        raise FileNotFoundError(f"Speech file not found: {speech_path}")

    print("Creating vectorstore from speech file (this may take a little while)...")
    loader = TextLoader(speech_path, encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n"]
    )
    docs_split = splitter.split_documents(docs)

    vectordb = Chroma.from_documents(
        documents=docs_split,
        embedding=embed,
        persist_directory=persist_dir
    )
    try:
        vectordb.persist()
    except Exception:
        pass

    print(f"Vectorstore created and saved to '{persist_dir}'.")
    return vectordb

def build_qa_chain(vectordb):

   # Build a RetrievalQA chain using Ollama (local) and the provided vectorstore's retriever.
    # Ollama wrapper - this uses the local Ollama daemon; ensure model pulled
    try:
        llm = Ollama(model=OLLAMA_MODEL)
    except Exception as e:
        print("Failed to initialize Ollama LLM wrapper. Ensure Ollama is installed and 'mistral' model is pulled.")
        print("Example: ollama pull mistral")
        raise e

    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": RETRIEVAL_K})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa_chain

def run_cli(qa_chain):
    #Simple interactive loop. Type a question, get an answer based only on the speech.txt content.

    print("\nAmbedkarGPT — Q&A (answers based ONLY on speech.txt)")
    print("Type a question and press Enter. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            query = input("Q: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        try:
            answer = qa_chain.run(query)                                #this is where ollama LLM works on retrieving answers from user queries input
        except Exception as e:
            print(f"[Error while generating answer] {e}")
            continue

        print("\nA:", answer.strip(), "\n" + "-"*60)                            #Note :- strip removes any extra spaces

def main():

    if not os.path.exists(SPEECH_FILE):
        print(f"Error: '{SPEECH_FILE}' not found in project root. Add it and re-run.")
        return

    vectordb = create_or_load_vectorstore(SPEECH_FILE, CHROMA_PERSIST_DIR)

    # QA chain and start CLI
    qa_chain = build_qa_chain(vectordb)
    run_cli(qa_chain)

if __name__ == "__main__":
    main()
