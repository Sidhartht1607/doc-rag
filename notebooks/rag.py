from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


@lru_cache(maxsize=1)
def _vectorstore(vector_name):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.load_local(vector_name, embeddings, allow_dangerous_deserialization=True)


def retrieve(query, k=4):
    """Return the top-k matching chunks as plain text."""
    docs = _vectorstore(vector_name="faiss_index").similarity_search(query, k=k)
    return "\n\n".join(d.page_content for d in docs)


def rag_prompt(query):
    return f"Answer using only this context:\n{retrieve(query)}\n\nQuestion: {query}"
