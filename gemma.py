from PyPDF2 import PdfReader
import requests
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai
import pdfplumber

def extract_text_from_pdf(pdf_path):
    full_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # 테이블 추출 시도
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        for row in table:
                            full_text += " ".join(str(cell) for cell in row if cell) + "\n"
                
                # 텍스트 추출
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
    except Exception as e:
        print(f"PDF 처리 중 오류 발생: {e}")
    return full_text

def extract_text_from_pdfplumber(pdf_path):
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            full_text += page.extract_text() or ""
    return full_text

def split_text(text, chunk_size=1000, overlap=200):
    chunks = []
    sentences = text.split('.')
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + "."
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + "."
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
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
            "model": "gemma:7b",  # ollama run mistral 에서 실행된 모델
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

def answer_question(question, chunks, index, embedding_model, top_k=3):
    q_emb = embedding_model.encode([question])
    D, I = index.search(q_emb, top_k)
    relevant_chunks = [chunks[i] for i in I[0]]
    
    context = "\n\n".join(relevant_chunks)
    prompt = (
        f"당신은 문서 분석 전문가입니다. 다음 문서를 주의 깊게 읽고 질문에 답변해주세요.\n\n"
        f"문서 내용:\n{context}\n\n"
        f"질문: {question}\n\n"
        f"지시사항:\n"
        f"1. 문서에 명시된 내용만을 바탕으로 답변하세요.\n"
        f"2. 문서에 없는 내용은 추측하지 마세요.\n"
        f"3. 답변은 한국어로 작성하세요.\n"
        f"4. 가능한 한 자세하고 정확하게 답변하세요.\n\n"
        f"답변:"
    )
    
    return query_ollama(prompt)

question = "시스템구성에 대해 설명해줘"
answer = answer_question(question, chunks, index, embedding_model)
print("▶ 답변:", answer)