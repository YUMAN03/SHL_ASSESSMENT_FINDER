# main.py
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_pinecone import PineconeEmbeddings
from dotenv import load_dotenv
import os
import re

load_dotenv()

app = FastAPI()

# Allow frontend (Streamlit) to access this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to Streamlit's IP in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API keys ---
pinecone_api_key = os.getenv("pinecone_api_key")
groq_api_key = os.getenv("groq_api_key")

# --- LangChain setup ---
llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model='llama3-70b-8192')
=
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("shldb2")
embeddings = PineconeEmbeddings(api_key=pinecone_api_key, model='multilingual-e5-large')
vectorstore = PineconeVectorStore(index=index, embedding=embeddings)

retriever = vectorstore.as_retriever(
    search_type='similarity_score_threshold',
    search_kwargs={'k': 20, 'score_threshold': 0.7},
)

template = '''
List of assesments : \n
{context}
From the list of 20 assessments I've identified, select the top 1-5 options that best match technical skills (prioritized) followed by soft skills. Focus especially on programming languages and technical frameworks.

For each assessment, provide ONLY the following information in this exact format:
- Title: [exact title]
- URL: [complete URL]
- Remote Testing: [Yes/No]
- Adaptive/IRT: [Yes/No] 
- Duration: [specific time OR "Not specified" if not mentioned]
- Test Type: [extract this directly from the description after "test type"]

Important instructions:
1. Return exactly 5 assessments total
2. The duration listed must not exceed what's actually specified
3. Present ONLY the assessment list - no introduction, explanation, or conclusion
4. Sort by technical skill match first, then soft skill relevance
Answer for the following query: \n
{query}
'''

prompt = PromptTemplate.from_template(template)
llm_chain = prompt | llm

def parse_assessments(text):
    pattern = r"- Title: (.*?)\n- URL: (.*?)\n- Remote Testing: (.*?)\n- Adaptive/IRT: (.*?)\n- Duration: (.*?)\n- Test Type: (.*?)\n"
    matches = re.findall(pattern, text)
    assessments = []
    for match in matches:
        assessments.append({
            "title": match[0],
            "url": match[1],
            "remote": match[2],
            "adaptive": match[3],
            "duration": match[4],
            "test_type": match[5]
        })
    return assessments

@app.get("/query")
def query_llm(q: str = Query(..., description="User query text")):
    docs = retriever.invoke(q)
    context = "\n\n".join(doc.page_content for doc in docs)
    response = llm_chain.invoke({"context": context, "query": q})
    assessments = parse_assessments(response.content)
    return {"query": q, "results": assessments}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
