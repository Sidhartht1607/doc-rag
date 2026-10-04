"""Document Q&A: hybrid retrieval, rerank, and a citing, abstaining agent."""
import os

# faiss and torch each bring an OpenMP runtime. On macOS arm64 the two crashed (segfault, exit 139)
# whenever embeddings were computed after faiss/pymupdf were imported. Single-threaded OpenMP fixed it
# in every test (3/3 runs vs 0/3 without). This must be set before those libraries load, and the
# cost is negligible at this index size. Override by exporting OMP_NUM_THREADS yourself.
os.environ.setdefault("OMP_NUM_THREADS", "1")
