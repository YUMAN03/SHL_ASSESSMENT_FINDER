import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_pinecone import PineconeEmbeddings
from pinecone_text.sparse import BM25Encoder
import re
from dotenv import load_dotenv
import os

load_dotenv()

# --- API Keys ---
pinecone_api_key = os.getenv("pinecone_api_key")
groq_api_key = os.getenv("groq_api_key")

# --- LLM Setup ---
llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model='llama3-70b-8192')

# --- Pinecone Setup ---
bm25_encoder = BM25Encoder().default()
pc = Pinecone(api_key=pinecone_api_key)
index_name = 'shldb2'
index = pc.Index(index_name)
embeddings = PineconeEmbeddings(api_key=pinecone_api_key, model='multilingual-e5-large')
vectorstore = PineconeVectorStore(index=index, embedding=embeddings)

retriever = vectorstore.as_retriever(
    search_type='similarity_score_threshold',
    search_kwargs={'k': 20, 'score_threshold': 0.7},
)

# --- Prompt Template ---
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

# --- Streamlit UI ---
st.set_page_config(page_title="Assessment Finder", page_icon="🧠")
st.title("🧠 SHL ASSESSMENT FINDER")

query = st.text_area("🔍 Enter your query", height=150)


if st.button("Search"):
    if query.strip() == "":
        st.warning("Please enter a query.")
    else:
        with st.spinner("Searching and analyzing assessments..."):
            # Step 1: Retrieve relevant documents
            docs = retriever.invoke(query)
            context = "\n\n".join(doc.page_content for doc in docs)

            # Step 2: Generate final response
            response = llm_chain.invoke({"context": context, "query": query})
            st.subheader("✅ Top Matching Assessments")
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

            # Parse response into structured list
            assessments = parse_assessments(response.content)

            # Display with formatting
            for i, a in enumerate(assessments, start=1):
                with st.container():
                    st.markdown(f"### 📘 {i}. {a['title']}")
                    st.markdown(f"- 🔗 [Assessment Link]({a['url']})")
                    st.markdown(f"- 🧪 **Remote Testing:** {a['remote']}")
                    st.markdown(f"- 📊 **Adaptive/IRT:** {a['adaptive']}")
                    st.markdown(f"- ⏱️ **Duration:** {a['duration']}")
                    st.markdown(f"- 📚 **Test Type:** {a['test_type']}")
                    st.markdown("---")

