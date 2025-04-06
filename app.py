# app.py
import streamlit as st
import requests

st.set_page_config(page_title="SHL Assessment Finder", page_icon="🧠")
st.title("🧠 SHL ASSESSMENT FINDER")

query = st.text_area("🔍 Enter your query", height=150)

if st.button("Search"):
    if query.strip() == "":
        st.warning("Please enter a query.")
    else:
        with st.spinner("Searching and analyzing assessments..."):
            try:
                api_url = "https://shlassessmentfinder-production.up.railway.app/query"
                response = requests.get(api_url, params={"q": query})
                if response.status_code == 200:
                    data = response.json()
                    assessments = data["results"]

                    st.subheader("✅ Top Matching Assessments")
                    for i, a in enumerate(assessments, start=1):
                        with st.container():
                            st.markdown(f"### 📘 {i}. {a['title']}")
                            st.markdown(f"- 🔗 [Assessment Link]({a['url']})")
                            st.markdown(f"- 🧪 **Remote Testing:** {a['remote']}")
                            st.markdown(f"- 📊 **Adaptive/IRT:** {a['adaptive']}")
                            st.markdown(f"- ⏱️ **Duration:** {a['duration']}")
                            st.markdown(f"- 📚 **Test Type:** {a['test_type']}")
                            st.markdown("---")
                else:
                    st.error(f"API error: {response.status_code}")
            except Exception as e:
                st.error(f"API call failed: {e}")
