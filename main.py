from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_pinecone import PineconeEmbeddings
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import json
import os
import re

load_dotenv()

app = FastAPI()

# Allow frontend (Streamlit) to access this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API keys ---
pinecone_api_key = os.getenv("pinecone_api_key")
groq_api_key = os.getenv("groq_api_key")

# --- LangChain setup ---
llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model='llama3-70b-8192')
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("shldb2")
embeddings = PineconeEmbeddings(api_key=pinecone_api_key, model='multilingual-e5-large')
vectorstore = PineconeVectorStore(index=index, embedding=embeddings)

retriever = vectorstore.as_retriever(
    search_type='similarity_score_threshold',
    search_kwargs={'k': 20, 'score_threshold': 0.8},
)

# Template for the frontend query endpoint
frontend_template = '''
List of assessments:
{context}
From the list of assessments I've identified, select the top options (up to 10 or fewer if there are less matches) . Focus especially on programming languages and technical frameworks.

For each assessment, provide ONLY the following information in this exact format:
- Title: [exact title]
- URL: [complete URL]
- Remote Testing: [Yes/No]
- Adaptive/IRT: [Yes/No] 
- Duration: [specific time OR "Not specified" if not mentioned]
- Test Type: [extract this directly from the description after "test type"]

Important instructions:
1. Return between 1-10 relevant assessments, with most relevant first
2. The duration listed must not exceed what's actually specified
3. Present ONLY the assessment list - no introduction, explanation, or conclusion
4. If there are technical skills(like programming languages and technical frameworks)mentioned then prioritize them.
5. Otherwise, prioritize by soft skils. 

Answer for the following query:
{query}
'''

# Template for the new /recommend endpoint
api_template = '''
List of assessments:
{context}

Based on the query, recommend appropriate assessments that would be useful for evaluating candidates.
Return exactly 10 assessments (or fewer if there aren't enough matches).

For each assessment, extract the following attributes in JSON format:
- url: The complete URL to access the assessment
- adaptive_support: "Yes" if the assessment is adaptive, "No" otherwise
- description: A brief description of what the assessment evaluates
- duration: The duration in minutes (as an integer)
- remote_support: "Yes" if the assessment can be taken remotely, "No" otherwise
- test_type: An array of strings representing categories (e.g., ["Knowledge & Skills"], ["Competencies", "Personality & Behaviour"])

Important instructions:
1. Return between 1-10 relevant assessments, with most relevant first
2. The duration listed must not exceed what's actually specified
3. Present ONLY the assessment list - no introduction, explanation, or conclusion
4. If there are technical skills(like programming languages and technical frameworks)mentioned then prioritize them.
6. Focus especially on programming languages and technical frameworks.
5. Otherwise, prioritize by soft skils. 

Query: {query}
'''

frontend_prompt = PromptTemplate.from_template(frontend_template)
api_prompt = PromptTemplate.from_template(api_template)

frontend_chain = frontend_prompt | llm
api_chain = api_prompt | llm

# Request model for the recommend endpoint
class RecommendRequest(BaseModel):
    query: str

# Function to parse assessments from the frontend format
def parse_frontend_assessments(text):
    pattern = r"- Title: (.*?)\n- URL: (.*?)\n- Remote Testing: (.*?)\n- Adaptive/IRT: (.*?)\n- Duration: (.*?)\n- Test Type: (.*?)(?:\n|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    assessments = []
    for match in matches:
        assessments.append({
            "title": match[0].strip(),
            "url": match[1].strip(),
            "remote": match[2].strip(),
            "adaptive": match[3].strip(),
            "duration": match[4].strip(),
            "test_type": match[5].strip()
        })
    return assessments[:10]  # Ensure we return at most 10 assessments

# Function to parse LLM response into structured assessment data for the API endpoint
def parse_api_assessments(response_text):
    try:
        # Extract JSON-like structures from the response text
        import json
        import re
        
        # Start by looking for JSON blocks
        json_pattern = r'\{.*?\}'
        
        # Try to find and parse assessments
        assessments = []
        
        # Look for structured data in LLM response
        matches = re.findall(r'url": "([^"]+)".*?"adaptive_support": "([^"]+)".*?"description": "([^"]+)".*?"duration": (\d+).*?"remote_support": "([^"]+)".*?"test_type": \[(.*?)\]', 
                             response_text, re.DOTALL)
        
        for match in matches:
            url, adaptive, desc, duration, remote, test_types = match
            # Parse the test_types string into a list
            types_list = [t.strip().strip('"\'') for t in test_types.split(',')]
            
            assessments.append({
                "url": url,
                "adaptive_support": adaptive,
                "description": desc,
                "duration": int(duration),
                "remote_support": remote,
                "test_type": types_list
            })
        
        # If no structured data found, try parsing from raw text
        if not assessments:
            # Find assessment blocks
            assessment_blocks = re.split(r'\d+\.\s+', response_text)[1:] 
            
            for block in assessment_blocks:
                url_match = re.search(r'URL:?\s*(https?://\S+)', block)
                adaptive_match = re.search(r'[Aa]daptive:?\s*([Yy]es|[Nn]o)', block)
                desc_match = re.search(r'[Dd]escription:?\s*(.+?)(?:\n|$)', block)
                duration_match = re.search(r'[Dd]uration:?\s*(\d+)', block)
                remote_match = re.search(r'[Rr]emote:?\s*([Yy]es|[Nn]o)', block)
                test_type_match = re.search(r'[Tt]est [Tt]ype:?\s*(.+?)(?:\n|$)', block)
                
                if url_match:
                    url = url_match.group(1)
                    adaptive = adaptive_match.group(1) if adaptive_match else "No"
                    desc = desc_match.group(1) if desc_match else "Not specified"
                    duration = int(duration_match.group(1)) if duration_match else 30
                    remote = remote_match.group(1) if remote_match else "No"
                    
                    if test_type_match:
                        test_types = test_type_match.group(1)
                        if '[' in test_types:
                            test_type = json.loads(test_types)
                        else:
                            test_type = [t.strip() for t in test_types.split(',')]
                    else:
                        test_type = ["Not specified"]
                    
                    assessments.append({
                        "url": url,
                        "adaptive_support": adaptive.capitalize(),
                        "description": desc,
                        "duration": duration,
                        "remote_support": remote.capitalize(),
                        "test_type": test_type
                    })
        
        # Ensure we only return up to 10 assessments
        return assessments[:10]
    except Exception as e:
        # Fallback if parsing fails
        print(f"Error parsing assessments: {e}")
        return []


@app.get("/health")
def health_check():
    return JSONResponse(content={"status": "healthy"},media_type="application/json")

@app.post("/recommend")
async def recommend_assessments(request: RecommendRequest):
    try:
        docs = retriever.invoke(request.query)
        context = "\n\n".join(doc.page_content for doc in docs)
        response = api_chain.invoke({"context": context, "query": request.query})
        assessments = parse_api_assessments(response.content)
        
        if not assessments:
            assessments = [{
                "url": "https://www.shl.com/solutions/products/product-catalog/view/general-aptitude/",
                "adaptive_support": "No",
                "description": "General aptitude assessment for basic skills evaluation",
                "duration": 30,
                "remote_support": "Yes",
                "test_type": ["Knowledge & Skills"]
            }]
        
        return JSONResponse(content={"recommended_assessments": assessments}, media_type="application/json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

# 3. Original Query Endpoint (For the Streamlit frontend)
@app.get("/query")
def query_llm(q: str = Query(..., description="User query text")):
    try:
        # Get relevant documents from vector store
        docs = retriever.invoke(q)
        context = "\n\n".join(doc.page_content for doc in docs)
        
        # Generate recommendations using LLM
        response = frontend_chain.invoke({"context": context, "query": q})
        
        # Parse the response to extract assessment data
        assessments = parse_frontend_assessments(response.content)
        
        # Ensure at least one assessment is returned
        if not assessments:
            # If no assessments were found, provide a default one
            assessments = [{
                "title": "General Aptitude Assessment",
                "url": "https://www.shl.com/solutions/products/product-catalog/view/general-aptitude/",
                "remote": "Yes",
                "adaptive": "No",
                "duration": "30 minutes",
                "test_type": "Knowledge & Skills"
            }]
            
        return {"query": q, "results": assessments}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
    
@app.get("/")
def index():
    return {
        "message": "Welcome to the SHL Assessment Recommendation API.",
        "endpoints": {
            "GET /health": "Returns health status of the API",
            "POST /recommend": "Returns assessment recommendations. Pass JSON body: { \"query\": \"your job description\" }",
            "GET /query?q=your+query": "Returns assessments for Streamlit frontend"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
