import asyncio
import logging
import numpy as np
import pymupdf4llm
import faiss
from sentence_transformers import SentenceTransformer
from typing import List
from pydantic_ai import Agent
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Set log level to WARNING for all other modules
for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.WARNING)

# Set our logger to DEBUG
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

load_dotenv()

agent = Agent('openai:gpt-4o')


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_text_from_pdf(pdf_path: str) -> str:
    md_text = pymupdf4llm.to_markdown(pdf_path)
    return md_text

def clean_text(text: str) -> str:
    """Clean text by replacing newlines with spaces and normalizing whitespace."""
    # Replace newlines with spaces
    text = text.replace('\n', ' ')
    # Replace multiple spaces with a single space
    text = ' '.join(text.split())
    return text.strip()


def chunk_texts(text: str, chunk_size=500, overlap=100) -> List[str]:
    # First clean the entire text to remove newlines and extra spaces
    cleaned_text = clean_text(text)
    
    chunks = []
    start = 0
    while start < len(cleaned_text):
        end = min(start + chunk_size, len(cleaned_text))
        # Ensure we don't cut words in the middle (find the last space in the chunk)
        if end < len(cleaned_text):
            last_space = cleaned_text.rfind(' ', start, end)
            if last_space > start + chunk_size * 0.8:
                end = last_space
        
        chunk = cleaned_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap if end - overlap > start else end
        
    return chunks

def build_faiss_index(chunks: List[str]):
    embeddings = embedding_model.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, chunks

# 根據使用者查詢做相似度檢索
async def search_faiss_index(index: faiss.IndexFlatL2, chunks: List[str], query: str, top_k=3):
    hyde_prompt = f"""
Write a paragraph of hypothetical document about question as you can.

Only return the paper content without any other information, ie. leading text and so on.

Question: {query}

"""
    response = await agent.run(hyde_prompt)
    logger.debug(f"Generated hypothetical document for query: {response.output}")
    query_embedding = embedding_model.encode([response.output])
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]


async def answer_query_with_context(query: str, context_chunks: List[str]) -> str:
    context = "\n\n".join(context_chunks)
    prompt = f"""
You are a DeFi expert, please answer the question based on the following content, if content's information is not enough, please answer "I don't know":

Content:
{context}

Question: {query}

Please answer in Chinese."""

    response = await agent.run(prompt)
    return response.output

def prepare_database(pdf_paths: list[str]) -> tuple[faiss.IndexFlatL2, list[str]]:
    """
    Prepare the FAISS index and chunk database from PDF files.
    
    Args:
        pdf_paths: List of paths to PDF files to process
        
    Returns:
        tuple: (faiss_index, chunk_database)
    """
    all_chunks = []
    for path in pdf_paths:
        logger.info(f"Processing PDF: {path}")
        texts = extract_text_from_pdf(path)
        chunks = chunk_texts(texts)
        all_chunks.extend(chunks)

    logger.info(f"Processed {len(all_chunks)} text chunks")
    
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Sample chunks:")
        for i, chunk in enumerate(all_chunks[:5]):
            logger.debug(f"Chunk {i+1} (length: {len(chunk)}): {chunk}")

    return build_faiss_index(all_chunks)

# 主程式
async def main():
    pdf_paths = [
        "src/data/How-to-DeFi-Beginner.pdf",
        "src/data/How-to-DeFi-Advanced.pdf"
    ]
    
    index, chunk_db = prepare_database(pdf_paths)

    while True:
        user_query = input("請輸入你的問題（輸入 q 離開）：")
        if user_query.lower() == 'q':
            break
        context = await search_faiss_index(index, chunk_db, user_query)
        logger.info(f"Retrieved {len(context)} relevant context chunks for query")
        for i, c in enumerate(context):
            logger.debug(f"Context {i+1} (length: {len(c)}): {c}")
        
        logger.info("Generating answer...")
        answer = await answer_query_with_context(user_query, context)
        logger.info("Answer generated")
        print("\n🔎 回答：\n", answer)

if __name__ == "__main__":
    asyncio.run(main())