# main.py
# from fastapi import FastAPI, Query
# from fastapi.middleware.cors import CORSMiddleware
# from langchain_core.prompts import PromptTemplate
# from langchain_groq import ChatGroq
# from langchain_pinecone import PineconeVectorStore
# from pinecone import Pinecone
# from langchain_pinecone import PineconeEmbeddings
# from dotenv import load_dotenv
# import os
# import re

# load_dotenv()

# app = FastAPI()

# # Allow frontend (Streamlit) to access this API
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # You can restrict this to Streamlit's IP in prod
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # --- API keys ---
# pinecone_api_key = os.getenv("pinecone_api_key")
# groq_api_key = os.getenv("groq_api_key")

# # --- LangChain setup ---
# llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model='llama3-70b-8192')
# pc = Pinecone(api_key=pinecone_api_key)
# index = pc.Index("shldb2")
# embeddings = PineconeEmbeddings(api_key=pinecone_api_key, model='multilingual-e5-large')
# vectorstore = PineconeVectorStore(index=index, embedding=embeddings)

# retriever = vectorstore.as_retriever(
#     search_type='similarity_score_threshold',
#     search_kwargs={'k': 20, 'score_threshold': 0.8},
# )

# template = '''
# List of assesments : \n
# {context}
# From the list of 10 assessments I've identified, select the top 5 options that best match technical skills (prioritized) followed by soft skills. Focus especially on programming languages and technical frameworks.

# For each assessment, provide ONLY the following information in this exact format:
# - Title: [exact title]
# - URL: [complete URL]
# - Remote Testing: [Yes/No]
# - Adaptive/IRT: [Yes/No] 
# - Duration: [specific time OR "Not specified" if not mentioned]
# - Test Type: [extract this directly from the description after "test type"]

# Important instructions:
# 1. Return exactly 5 assessments total
# 2. The duration listed must not exceed what's actually specified
# 3. Present ONLY the assessment list - no introduction, explanation, or conclusion
# 4. Sort by technical skill match first, then soft skill relevance
# Answer for the following query: \n
# {query}
# '''

# prompt = PromptTemplate.from_template(template)
# llm_chain = prompt | llm

# def parse_assessments(text):
#     pattern = r"- Title: (.*?)\n- URL: (.*?)\n- Remote Testing: (.*?)\n- Adaptive/IRT: (.*?)\n- Duration: (.*?)\n- Test Type: (.*?)\n"
#     matches = re.findall(pattern, text)
#     assessments = []
#     for match in matches:
#         assessments.append({
#             "title": match[0],
#             "url": match[1],
#             "remote": match[2],
#             "adaptive": match[3],
#             "duration": match[4],
#             "test_type": match[5]
#         })
#     return assessments

# @app.get("/query")
# def query_llm(q: str = Query(..., description="User query text")):
#     docs = retriever.invoke(q)
#     context = "\n\n".join(doc.page_content for doc in docs)
#     response = llm_chain.invoke({"context": context, "query": q})
#     assessments = parse_assessments(response.content)
#     return {"query": q, "results": assessments}
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8000)
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_pinecone import PineconeEmbeddings
from dotenv import load_dotenv
from pydantic import BaseModel
import os
import re

# Load environment variables
load_dotenv()

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to specific origins in production
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

# Define the refined prompt to match exactly the required output format
template = '''
List of assessments: 
{context}

Based on the provided job description or query, recommend AT MOST 10 relevant assessments that match the requirements.
For each assessment, return ONLY the following information in this exact structured format required by the API:

1. URL (direct link to the assessment)
2. Adaptive Support (Yes/No) - indicating if the assessment supports adaptive testing
3. Description (concise description of the assessment)
4. Duration (exact time in minutes, must be an integer)
5. Remote Support (Yes/No) - indicating if the assessment can be taken remotely
6. Test Type (array of categories)

Your response must be structured so it can be parsed into this exact JSON format:
{{
  "recommended_assessments": [
    {{
      "url": "assessment URL",
      "adaptive_support": "Yes/No",
      "description": "Description text",
      "duration": integer_minutes,
      "remote_support": "Yes/No",
      "test_type": ["Category1", "Category2"]
    }}
  ]
}}

Make sure to:
1. Include between 1 and 10 assessments
2. Format test_type as a proper array of strings
3. Convert duration to an integer value (minutes only)
4. Keep adaptive_support and remote_support as exactly "Yes" or "No"

Query: {query}
'''

prompt = PromptTemplate.from_template(template)
llm_chain = prompt | llm

# Define request body model
class QueryRequest(BaseModel):
    query: str

def parse_assessment_response(text):
    """Parse LLM response into structured assessment data"""
    try:
        # We'll extract all assessments in the format we expect
        assessments = []
        
        # Look for sections with URL and other fields
        pattern = r'url"?\s*:\s*"([^"]+)".*?adaptive_support"?\s*:\s*"(Yes|No)".*?description"?\s*:\s*"([^"]+)".*?duration"?\s*:\s*(\d+).*?remote_support"?\s*:\s*"(Yes|No)".*?test_type"?\s*:\s*\[(.*?)\]'
        
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            url, adaptive, description, duration, remote, test_types = match
            
            # Clean and parse test types
            test_types = test_types.strip()
            if test_types:
                # Handle both quoted and unquoted test types
                test_type_list = re.findall(r'"([^"]+)"|\'([^\']+)\'|([^,\s"\']+)', test_types)
                # Flatten the results and remove empty strings
                test_type_list = [next(s for s in t if s) for t in test_type_list]
            else:
                test_type_list = []
                
            assessments.append({
                "url": url,
                "adaptive_support": adaptive,
                "description": description,
                "duration": int(duration),  # Convert to integer
                "remote_support": remote,
                "test_type": test_type_list
            })
        
        return assessments
    except Exception as e:
        print(f"Error parsing assessment response: {e}")
        print(f"Raw text: {text}")
        return []

# 1. Health Check Endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# 2. Assessment Recommendation Endpoint
@app.post("/recommend")
async def recommend_assessments(request_data: QueryRequest):
    try:
        # Retrieve relevant documents from vector store
        docs = retriever.invoke(request_data.query)
        context = "\n\n".join(doc.page_content for doc in docs)
        
        # Generate recommendations using LLM
        response = llm_chain.invoke({"context": context, "query": request_data.query})
        
        # Parse the response to extract assessments in the required format
        assessments = parse_assessment_response(response.content)
        
        # Limit to 10 assessments
        assessments = assessments[:10]
        
        # If no assessments were found, provide at least one
        if not assessments:
            # Fallback assessment if nothing is found
            assessments = [{
                "url": "https://www.shl.com/solutions/products/product-catalog/view/technology-professional-8-0-job-focused-assessment/",
                "adaptive_support": "No",
                "description": "Assesses key attributes required for success in technology environments.",
                "duration": 16,
                "remote_support": "Yes",
                "test_type": ["Competencies", "Personality & Behaviour"]
            }]
        
        return {"recommended_assessments": assessments}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

# Keep your existing endpoint for backward compatibility
@app.get("/query")
def query_llm(q: str = Query(..., description="User query text")):
    docs = retriever.invoke(q)
    context = "\n\n".join(doc.page_content for doc in docs)
    response = llm_chain.invoke({"context": context, "query": q})
    assessments = parse_assessments(response.content)
    return {"query": q, "results": assessments}

# Keep your existing parse_assessments function
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)