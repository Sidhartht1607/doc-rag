"""
Evaluation set for document.pdf (AWS Prescriptive Guidance:
"Writing best practices to optimize RAG applications").

Each item:
  id            - stable identifier
  question      - what the user asks
  answer        - reference answer, written from the PDF text
  pages         - 0-indexed PDF pages that contain the answer. These match
                  chunk.metadata["page"] from PyMuPDFLoader (page 0 = cover).
                  Printed page number in the PDF = index - 3 (roughly).
  evidence      - short phrase copied from the PDF. Chunk boundaries change when
                  you tune chunk_size/overlap, so check "evidence in chunk text"
                  rather than a chunk id when scoring retrieval.
  answerable    - False means the PDF does NOT contain the answer; the agent
                  should abstain instead of making something up.
"""

EVAL_SET = [
    # ---------- Answerable: RAG basics ----------
    {
        "id": "q01",
        "question": "What are the two stages of RAG?",
        "answer": "In the first stage, a retrieval model finds relevant documents or passages "
                  "based on the query. In the second stage, the retrieved information and the "
                  "original query are fed into an LLM as a materialized prompt template, and the "
                  "LLM generates the response.",
        "pages": [7],
        "evidence": "RAG employs a two-stage process",
        "answerable": True,
    },
    {
        "id": "q02",
        "question": "What are the steps in the RAG application workflow, from the user's query to the response?",
        "answer": "The user submits a query; the app queries a vector database of knowledge sources; "
                  "it retrieves relevant information by semantic similarity; it augments the original "
                  "prompt with the retrieved context and sends it to the LLM endpoint; the LLM generates "
                  "a response and returns it to the app; the app returns the response to the user.",
        "pages": [6, 7],
        "evidence": "The RAG application augments the original prompt with the retrieved context",
        "answerable": True,
    },
    {
        "id": "q03",
        "question": "What is a vector database used for in generative AI?",
        "answer": "It stores and manages vector representations of documents, queries or other objects "
                  "and retrieves them efficiently, supporting fast, scalable operations such as semantic "
                  "search and similarity matching.",
        "pages": [7],
        "evidence": "a vector database is a database that stores and manages vector representations",
        "answerable": True,
    },
    {
        "id": "q04",
        "question": "Which data structures do vector databases use to find nearest neighbors quickly?",
        "answer": "Specialized data structures such as Hierarchical Navigable Small World (HNSW) graphs "
                  "or K-Nearest Neighbors (KNN) algorithms.",
        "pages": [7],
        "evidence": "Hierarchical Navigable Small World (HNSW) graphs or K-Nearest Neighbors (KNN)",
        "answerable": True,
    },
    {
        "id": "q05",
        "question": "What is semantic search and how does it differ from keyword matching?",
        "answer": "Semantic search improves relevance by understanding the intent and context of the "
                  "query rather than just matching keywords. It compares the vector representations of "
                  "the query and the documents to find the most relevant matches.",
        "pages": [8],
        "evidence": "understanding the intent and context of the query, rather than just matching keywords",
        "answerable": True,
    },
    {
        "id": "q06",
        "question": "What is cosine similarity?",
        "answer": "A measure of similarity between two non-zero vectors that computes the cosine of the "
                  "angle between them. It is often used in semantic search to compare the direction of "
                  "vectors in high-dimensional space.",
        "pages": [8],
        "evidence": "Cosine similarity – A measure of similarity between two non-zero vectors",
        "answerable": True,
    },
    {
        "id": "q07",
        "question": "What does locality-sensitive hashing (LSH) do?",
        "answer": "It hashes similar vectors to the same or nearby buckets with high probability, which "
                  "allows approximate nearest-neighbor searches that can be faster than exact searches "
                  "in high-dimensional spaces.",
        "pages": [8],
        "evidence": "hashes similar vectors to the same or nearby buckets",
        "answerable": True,
    },
    {
        "id": "q08",
        "question": "What is context engineering, and how does it relate to agentic RAG?",
        "answer": "Optimizing the context delivery process at a system level is called context "
                  "engineering, and it is an essential part of agentic RAG architectures. In agentic RAG, "
                  "one or more additional LLMs reason and act on intake requests before the RAG execution, "
                  "enabling a multi-step information delivery process.",
        "pages": [3],
        "evidence": "Optimizing the context delivery process at a system level is called context engineering",
        "answerable": True,
    },
    {
        "id": "q09",
        "question": "Which external sources does the guide give as examples for RAG retrieval?",
        "answer": "Knowledge Bases for Amazon Bedrock, intelligent search systems such as Amazon Kendra, "
                  "and vector databases such as Amazon OpenSearch Service.",
        "pages": [5],
        "evidence": "Knowledge Bases for Amazon Bedrock, intelligent search systems such as Amazon Kendra",
        "answerable": True,
    },

    # ---------- Answerable: challenges in source data ----------
    {
        "id": "q10",
        "question": "What are the common challenges with raw documents that hurt RAG performance?",
        "answer": "Lack of structured formatting and metadata; informal and inconsistent language; "
                  "verbosity and redundancy; ambiguous terms and phrases; injection of graphic and "
                  "hyperlink elements; and lack of domain-specific knowledge or context.",
        "pages": [9],
        "evidence": "most common raw document challenges for an optimal RAG application",
        "answerable": True,
    },
    {
        "id": "q11",
        "question": "Why do graphics and hyperlinks in raw documents cause problems for RAG?",
        "answer": "They work well for humans but consume the retrieval token limit. Graphics and hyperlink "
                  "URLs get returned as part of the retrieval, so excerpts can be incomplete and key "
                  "information from subsequent paragraphs is missing.",
        "pages": [9],
        "evidence": "these elements can consume the retrieval token limit",
        "answerable": True,
    },

    # ---------- Answerable: best practices ----------
    {
        "id": "q12",
        "question": "Why should numbered lists be sequential?",
        "answer": "Proper numbering avoids confusion. Each list item should be numbered sequentially "
                  "without skipping numbers, which keeps the content clear and coherent.",
        "pages": [11],
        "evidence": "Ensure that each list item is numbered sequentially without skipping numbers",
        "answerable": True,
    },
    {
        "id": "q13",
        "question": "Why add transitions between list items?",
        "answer": "Transitions between items in a bulleted or numbered list help guide the LLM through "
                  "the content, for example 'After completing step 2, do...', connecting ideas and "
                  "improving the flow of information.",
        "pages": [11],
        "evidence": "Providing transitions between items in a bulleted or numbered list helps guide the LLM",
        "answerable": True,
    },
    {
        "id": "q14",
        "question": "Why does the guide recommend replacing tables with flat-level syntax?",
        "answer": "Tables are hard for RAG models to interpret because they require understanding a "
                  "two-dimensional structure. Documents are read left to right, so flat-level syntax or "
                  "bulleted lists let the information follow coherently without needing an extra "
                  "dimension, which models process more easily.",
        "pages": [11, 13],
        "evidence": "It can be challenging for RAG models to interpret tables",
        "answerable": True,
    },
    {
        "id": "q15",
        "question": "How should graphical information be handled for efficiency?",
        "answer": "Reduce the resolution of images, remove redundant images, and describe the content of "
                  "graphical elements in text. This adds meaningful context, avoids consuming tokens "
                  "unnecessarily and improves accessibility for RAG models.",
        "pages": [11],
        "evidence": "Reduce the resolution of images, remove redundant images",
        "answerable": True,
    },
    {
        "id": "q16",
        "question": "What is a session starter and why is it useful?",
        "answer": "A session starter is a sentence that transitions the reader into a process for a "
                  "common query, for example 'If you are looking to order software, follow the steps "
                  "below'. It creates high semantic matching, which helps the LLM build a cohesive response.",
        "pages": [11],
        "evidence": "add a session starter that transitions the reader into the process",
        "answerable": True,
    },
    {
        "id": "q17",
        "question": "How does adding a summary to each section improve RAG performance?",
        "answer": "A brief summary after each heading or subheading increases semantic coverage and "
                  "reinforces key points. This improves the accuracy of similarity search in the "
                  "embedding space, which improves the RAG application's performance.",
        "pages": [11, 12, 14],
        "evidence": "Add summarization to each section",
        "answerable": True,
    },
    {
        "id": "q18",
        "question": "Why should abbreviations be defined and context set in documents?",
        "answer": "LLMs are trained on broad internet data and usually lack context for an enterprise's "
                  "internal documents and abbreviations. Defining them helps the LLM understand "
                  "enterprise data, answer more accurately and avoid hallucinations or misinterpretations.",
        "pages": [12, 14],
        "evidence": "setting context, defining abbreviations, and avoiding or defining company-specific terminology",
        "answerable": True,
    },
    {
        "id": "q19",
        "question": "Why should large documents be split into smaller ones?",
        "answer": "Avoid indexing a large document with multiple subtopics. Dividing it into smaller, "
                  "self-contained documents with clear titles improves indexing and tagging.",
        "pages": [12],
        "evidence": "Restructure large documents into smaller documents for efficient tagging and indexing",
        "answerable": True,
    },
    {
        "id": "q20",
        "question": "How do headings and subheadings help a RAG application?",
        "answer": "Clear headings and subheadings help RAG models understand the structure and context "
                  "of the content, so they can navigate and extract relevant information better, which "
                  "improves the quality of generated responses.",
        "pages": [11, 13],
        "evidence": "Organizing your content with clear headings and subheadings improves readability",
        "answerable": True,
    },
    {
        "id": "q21",
        "question": "What does the guide recommend as the first step to begin optimizing documents for RAG?",
        "answer": "Conduct an audit of existing documents to identify areas that pose challenges to the "
                  "RAG application, such as lack of structure, ambiguous language or excessive use of "
                  "graphics, and prioritize documents that are frequently accessed or business critical.",
        "pages": [15],
        "evidence": "conducting an audit of your existing documents",
        "answerable": True,
    },
    {
        "id": "q22",
        "question": "Who is the intended audience of this guide?",
        "answer": "AI engineers, data scientists, data engineers and software developers building LLM "
                  "applications with one or more RAG components, who should be familiar with vector "
                  "databases and prompts for LLMs.",
        "pages": [3],
        "evidence": "AI engineers, data scientists, data engineers, or software developers",
        "answerable": True,
    },

    # ---------- Unanswerable: agent should abstain ----------
    {
        "id": "u01",
        "question": "What is the refund policy?",
        "answer": "NOT IN DOCUMENT. The agent should say it has no information on refunds.",
        "pages": [],
        "evidence": "",
        "answerable": False,
    },
    {
        "id": "u02",
        "question": "How much does Amazon Bedrock cost per 1,000 tokens?",
        "answer": "NOT IN DOCUMENT. The guide mentions Amazon Bedrock but gives no pricing.",
        "pages": [],
        "evidence": "",
        "answerable": False,
    },
    {
        "id": "u03",
        "question": "What chunk size and chunk overlap does the guide recommend for splitting documents?",
        "answer": "NOT IN DOCUMENT. It recommends splitting large documents into smaller self-contained "
                  "ones but gives no chunk size or overlap values.",
        "pages": [],
        "evidence": "",
        "answerable": False,
    },
    {
        "id": "u04",
        "question": "Give the step-by-step Terraform commands to deploy a RAG application on AWS.",
        "answer": "NOT IN DOCUMENT. The guide only lists a Terraform deployment guide as an external resource.",
        "pages": [],
        "evidence": "",
        "answerable": False,
    },
    {
        "id": "u05",
        "question": "Which embedding model does the guide say is the best for RAG?",
        "answer": "NOT IN DOCUMENT. The guide explains embeddings conceptually but names no recommended model.",
        "pages": [],
        "evidence": "",
        "answerable": False,
    },
]
