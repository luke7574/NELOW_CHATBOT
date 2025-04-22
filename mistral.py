from PyPDF2 import PdfReader
import requests
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text()
    return full_text

def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# 1) 임베딩 모델
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 2) Chunk별 벡터 생성
chunks = split_text(extract_text_from_pdf("C:/Users/user/AI/NELOW_WEB/nelow_web.pdf"))
embeddings = embedding_model.encode(chunks)

# 3) FAISS에 저장
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

def query_ollama(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",  # ollama run mistral 에서 실행된 모델
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

def answer_question(question, chunks, index, embedding_model, top_k=3):
    # 질문을 임베딩
    q_emb = embedding_model.encode([question])
    
    # 유사한 chunk 인덱스 찾기
    D, I = index.search(q_emb, top_k)
    relevant_chunks = [chunks[i] for i in I[0]]
    
    # 프롬프트 생성
    context = "\n\n".join(relevant_chunks)
    prompt = f"문맥: \n{context}\n\n질문: {question}\n\n답변:"
    
    # Ollama Mistral 사용
    return query_ollama(prompt)

question = "누수음 수집절차에 대해 설명해줘"
answer = answer_question(question, chunks, index, embedding_model)
print("▶ 답변:", answer)

