import os
from dotenv import load_dotenv

load_dotenv()
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse

# re-ranking for better result
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# metadata filtering
from qdrant_client.models import Filter, FieldCondition, MatchValue

# metadata extraction from LLM
from utils.schema import ChunkMetaData

# Configuration
COLLECTION_NAME = "industrial_docs"
EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash"

RERANKER_MODEL = "BAAI/bge-reranker-base"

# Initialize LLM
llm = ChatGoogleGenerativeAI(model=LLM_MODEL)

# Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

# Sparse embeddings
# 重写FastEmbedSparse类，在外面套一层分词器
from typing import List
import jieba
import re


class JiebaFastEmbedSparse(FastEmbedSparse):
    """在 FastEmbedSparse 外层包一个 jieba 分词预处理"""

    def _preprocess(self, text: str) -> str:
        seg = jieba.cut(text, cut_all=False)
        result = (" ").join(
            [
                token
                for token in seg
                if token.strip() and not re.match(r"^[\s\W]+$", token)
            ]
        )
        return result

    def embed_documents(self, texts: List[str]) -> List:
        tokenized = [self._preprocess(t) for t in texts]
        return super().embed_documents(tokenized)

    def embed_query(self, text: str) -> List:
        tokenized = self._preprocess(text)
        return super().embed_query(tokenized)


sparse_embeddings = JiebaFastEmbedSparse(model_name="Qdrant/bm25")

# Connect to existing collection
vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    sparse_embedding=sparse_embeddings,
    collection_name=COLLECTION_NAME,
    url="https://cb36196c-4fa0-4a84-af87-6a84b091f2b3.us-west-1-0.aws.cloud.qdrant.io:6333",
    api_key=os.getenv("QDRANT_API_KEY"),
    retrieval_mode=RetrievalMode.HYBRID,
)

vector_store.client.get_collection(collection_name=COLLECTION_NAME)

# ### Filter Extraction with LLM


def extract_filters(user_query: str):

    prompt = f"""
            从用户的问题中依据特定格式提取元数据信息

                <USER QUERY STARTS>
                {user_query}
                </USER QUERY ENDS>

                #### DESCRIPTIONS
                category:
                - 用户询问的问题类别,只要和工业自动化安全类型相关的,都必须归类为Industrial_Automation_Safety
                - 如果用户询问的问题和工业自动化安全没有任何关系,,则将问题归类为Other

                technology:
                - 用户询问的问题具体涉及的技术,如果没有合适的匹配项,则不要填写此项

                EXAMPLES:
                "使用控制阀的安全要求是什么？" -> {{"category": "Industrial_Automation_Safety", "technology": "Control_Valve"}}
                "使用温度变送器的注意事项？" -> {{"category": "Industrial_Automation_Safety", "technology": "Temperature_Transmitter"}}
                "什么是热力学第三定律?" -> {{"category": "Other"}}
                "生物学防治的注意事项" -> {{"category": "Other"}}

                Extract metadata based on the user query only.
            """

    structurerd_llm = llm.with_structured_output(ChunkMetaData)

    metadata = structurerd_llm.invoke(prompt)

    filters = metadata.model_dump(exclude_none=True)

    return filters


# ### Retrieval Functions
from langchain.tools import tool
@tool
def hybrid_search(query: str, k: int = 5):
    """
    Perform hybrid search (dense + sparse vectors).

    Args:
        query: Search query
        k: Number of results
        filters: Optional filters like {"category": "Industrial_Automation_Safety", "technology": "Temperature_Transmitter"}

    Returns:
        List of Document objects
    """

    filters = extract_filters(query)

    qdrant_filter = None

    if filters:
        condition = [
            FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value))
            for key, value in filters.items()
        ]

        qdrant_filter = Filter(must=condition)

    results = vector_store.similarity_search(query=query, k=k, filter=qdrant_filter)

    return results


# re-ranking for better result
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


def rerank_results(query: str, documents: list, top_k: int = 5):
    """
    Rerank documents using cross-encoder.

    Args:
        query: Search query
        documents: List of Document objects
        top_k: Number of top results to return

    Returns:
        List of (score, Document) tuples sorted by relevance
    """

    reranker = HuggingFaceCrossEncoder(
        model_name=RERANKER_MODEL, model_kwargs={"device": "cuda"}
    )

    query_doc_pairs = [(query, doc.page_content) for doc in documents]

    scores = reranker.score(query_doc_pairs)

    reranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)

    reranked = reranked[:top_k]
    return [rank[1] for rank in reranked]