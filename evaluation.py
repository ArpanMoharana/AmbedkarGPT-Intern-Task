import json
from pathlib import Path
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CORPUS_DIR = Path("corpus")
TEST_FILE = Path("test_dataset.json")
PERSIST_DIR = Path("chroma_minimal")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 550
CHUNK_OVERLAP = 100
K = 3
OUTPUT = Path("simple_results.json")


def build_vectordb():
    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    docs = []

    for p in sorted(CORPUS_DIR.glob("*.txt")):
        loader = TextLoader(str(p), encoding="utf-8")
        loaded = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n"],
        )
        chunks = splitter.split_documents(loaded)
        for chunk in chunks:
            # chunk.metadata may or may not exist depending on object; set safely
            if not hasattr(chunk, "metadata") or chunk.metadata is None:
                chunk.metadata = {}
            chunk.metadata["source"] = p.name
            docs.append(chunk)


    db = Chroma.from_documents(documents=docs, embedding=emb, persist_directory=str(PERSIST_DIR))
    try:
        db.persist()
    except Exception:
        pass
    return db


def compute_metrics(all_ret: List[List[str]], all_gold: List[List[str]]):
    n = len(all_ret)
    if n == 0:
        return {"hit_at_k": 0.0, "mrr": 0.0}

    hits = 0
    rr_sum = 0.0
    for ret, gold in zip(all_ret, all_gold):
        topk = ret[:K]

        # Hit@K
        if any(r in gold for r in topk):
            hits += 1

        # MRR: reciprocal rank of first correct doc in topk
        rr = 0.0
        for i, r in enumerate(topk, start=1):
            if r in gold:
                rr = 1.0 / i
                break
        rr_sum += rr

    return {"hit_at_k": hits / n, "mrr": rr_sum / n}


def main():
    # load test dataset (string -> json)
    test = json.loads(TEST_FILE.read_text(encoding="utf-8"))
    questions = test["test_questions"]

    db = build_vectordb()
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": K})

    all_ret = []
    all_gold = []
    per_q = []

    for q in questions:
        qid = q.get("id")
        qtext = q.get("question", "")
        gold = q.get("source_documents", []) or []

        docs = retriever.get_relevant_documents(qtext)
        ret = [d.metadata.get("source", "unknown") for d in docs]

        all_ret.append(ret)
        all_gold.append(gold)

        per_q.append({"id": qid, "question": qtext, "retrieved": ret, "gold": gold})

    metrics = compute_metrics(all_ret, all_gold)

    out = {"meta": {"k": K, "n_questions": len(questions)}, "metrics": metrics, "per_question": per_q}
    OUTPUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Done — saved {OUTPUT}. Hit@{K}={metrics['hit_at_k']:.3f}, MRR={metrics['mrr']:.3f}")


if __name__ == "__main__":
    main()
