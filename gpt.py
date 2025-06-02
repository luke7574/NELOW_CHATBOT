from PyPDF2 import PdfReader
import os

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

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1) 임베딩 모델
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 2) Chunk별 벡터 생성
chunks = split_text(extract_text_from_pdf("C:/Users/user/AI/NELOW_WEB/nelow_web.pdf"))
embeddings = embedding_model.encode(chunks)

# 3) FAISS에 저장
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))


import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def answer_question(question, chunks, index, embedding_model, top_k=3):
    # 질문을 임베딩
    q_emb = embedding_model.encode([question])
    
    # 유사한 chunk 인덱스 찾기
    D, I = index.search(q_emb, top_k)
    relevant_chunks = [chunks[i] for i in I[0]]
    
    # 프롬프트 생성
    context = "\n\n".join(relevant_chunks)
    prompt = f"문맥: \n{context}\n\n질문: {question}\n\n답변:"
    
    # OpenAI GPT로 응답 생성
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    
    return response.choices[0].message.content

question = "누수음 조회 및 정보 수정하는 방법법에 대해 설명해줘"
answer = answer_question(question, chunks, index, embedding_model)
print("▶ 답변:", answer)