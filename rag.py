import os
import json
import numpy as np

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder

from pinecone import Pinecone

from langchain_groq import ChatGroq

from typing import Literal
from pydantic import BaseModel, Field


# ============================================================
# Environment
# ============================================================

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


# ============================================================
# Embedding Manager
# ============================================================

class EmbeddingManager:
    """Handles embedding generation using sentence-transformers."""

    def __init__(self, model_name: str = "all-mpnet-base-v2"):

        self.model_name = model_name
        self.model = None

        self._load_model()

    def _load_model(self):

        try:
            print(f"Loading embedding model: {self.model_name}")

            self.model = SentenceTransformer(self.model_name)

            print(
                f"Loaded embedding model successfully. "
                f"Dimension: {self.model.get_embedding_dimension()}"
            )

        except Exception as e:

            print(
                f"Failed to load model {self.model_name}: {e}"
            )

            raise

    def generate_embeddings(self, text: list[str]) -> np.ndarray:

        if not self.model:
            raise ValueError("Model not loaded")

        print(f"Generating embedding for {len(text)} texts")

        embeddings = self.model.encode(
            text,
            show_progress_bar=False
        )

        return embeddings


# ============================================================
# Pinecone Connection
# ============================================================

pc = Pinecone(
    api_key=PINECONE_API_KEY
)

INDEX_NAME = "finsight"

index = pc.Index(INDEX_NAME)

print(f"Connected to Pinecone index: {INDEX_NAME}")


# ============================================================
# LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# ============================================================
# Search Query Schema
# ============================================================

class SearchQuery(BaseModel):

    intent: Literal[
        "question_answering",
        "filing_retrieval"
    ] = Field(
        description="The user's intended action."
    )

    company: Literal[
        "AAPL",
        "MSFT",
        "AMZN",
        "NVDA",
        "GOOGL"
    ] = Field(
        description="The company ticker."
    )

    filing_year: str | None = None

    query: str = Field(
        description="The user's actual financial question or request."
    )


# ============================================================
# Query Analyzer
# ============================================================
def analyze_query(user_question: str) -> SearchQuery:

    prompt = f"""
You are the query analyzer for FinSight AI.

Your job is to convert the user's question into a structured
retrieval query for an SEC financial-document RAG system.

Extract:

1. intent
2. company
3. filing_year
4. query

--------------------------------------------------
INTENT
--------------------------------------------------

Use:

- "question_answering"
  when the user wants an answer based on SEC filing information.

- "filing_retrieval"
  when the user explicitly wants to find or retrieve a filing/document.

--------------------------------------------------
COMPANY
--------------------------------------------------

Identify the company and return its stock ticker.

Allowed companies:

- AAPL
- MSFT
- AMZN
- NVDA
- GOOGL

--------------------------------------------------
FILING YEAR
--------------------------------------------------

This field represents the filing/document timeframe.

IMPORTANT:

If the user says:

- latest filing
- latest SEC filing
- most recent filing
- current filing
- newest filing

return:

"latest"

If the user explicitly mentions a year, return that year.

Examples:

"2025 filing" → "2025"

"2024 annual report" → "2024"

If no filing year is specified, return:

null

--------------------------------------------------
QUERY
--------------------------------------------------

Create a concise, focused financial retrieval query.

The query should contain the important financial concepts
needed to retrieve the relevant SEC document chunks.

DO NOT copy the entire user question.

For example:

User:
"Compare NVIDIA's revenue growth from its latest SEC filing
with its current stock price and summarize the latest relevant news."

Return a RAG query similar to:

"Return a RAG query similar to:

"NVIDIA fiscal year 2026 revenue fiscal year 2025 revenue annual revenue growth""

The RAG query should focus ONLY on information that belongs
in SEC filings.

Ignore live market information such as:

- current stock price
- latest news
- today's price
- market movements

Those will be handled by MCP.

--------------------------------------------------
OUTPUT
--------------------------------------------------

Return ONLY valid JSON.

Do not include markdown.
Do not include explanations.

User request:
{user_question}

JSON format:

{{
    "intent": "question_answering",
    "company": "NVDA",
    "filing_year": "latest",
    "query": "NVIDIA total revenue fiscal year 2026 fiscal year 2025 revenue growth"
}}
"""

    response = llm.invoke(prompt)

    data = json.loads(response.content)

    return SearchQuery(**data)



# ============================================================
# RAG Retriever
# ============================================================

class RAGRetriever:

    """Handles query-based retrieval from Pinecone."""

    def __init__(self, index, embedding_manager):

        self.index = index
        self.embedding_manager = embedding_manager

        print("Loading reranker: BAAI/bge-reranker-base")

        self.reranker = CrossEncoder(
        "BAAI/bge-reranker-base")

        print("Reranker loaded successfully")

    def retrieve(
        self,
        query: str,
        company: str,
        filing_year: str | None = None,
        top_k: int = 20,
        score_threshold: float = 0.0
    ):

        print(f"Retrieving documents for: '{query}'")
        print(f"Company: {company}")
        print(f"Filing year: {filing_year}")
        print(f"Top K: {top_k}")

        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embeddings(
            [query]
        )[0]


        # Search Pinecone
        filter_conditions = {
        "company": {
            "$eq": company
            }
        }


        results = self.index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True,
        filter=filter_conditions
        )

        # ========================================================
        # Re-ranking
        # ========================================================

        rerank_pairs = []

        for match in results.matches:

            document_text = match.metadata.get(
                "text",
                ""
            )

            rerank_pairs.append(
                (query, document_text)
            )

        # Calculate relevance scores
        rerank_scores = self.reranker.predict(
            rerank_pairs
        )

        # Combine Pinecone matches with reranker scores
        reranked_matches = []

        for match, score in zip(
            results.matches,
            rerank_scores
        ):

            reranked_matches.append(
                {
                    "match": match,
                    "rerank_score": float(score)
                }
            )

        # Sort by reranker score
        reranked_matches.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        # Keep only the best 5
        reranked_matches = reranked_matches[:15]

        print(
            f"Reranked {len(results.matches)} documents → "
            f"Top {len(reranked_matches)}"
        )

        # ========================================================
        # Convert to FinSight document format
        # ========================================================

        retrieved_docs = []

        for rank, item in enumerate(
            reranked_matches,
            start=1
        ):

            match = item["match"]
            rerank_score = item["rerank_score"]

            if match.score >= score_threshold:

                retrieved_docs.append({

                    "id": match.id,

                    "content": match.metadata.get(
                        "text"
                    ),

                    "metadata": match.metadata,

                    "similarity_score": match.score,

                    "rerank_score": rerank_score,

                    "rank": rank
                })

        print(f"Retrieved {len(retrieved_docs)} documents")
        return retrieved_docs

# ============================================================
# Initialize RAG Retriever
# ============================================================

embeddingmanager = EmbeddingManager()

rag_retriever = RAGRetriever(
    index=index,
    embedding_manager=embeddingmanager
)
