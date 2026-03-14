import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

@st.cache_resource(show_spinner=False)
def load_embeddings():
    # Points exactly to your local offline model
    model_path = r"D:\ai_agent_policy\models\all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(model_name=model_path)

@st.cache_resource(show_spinner=True)
def load_vector_store(_embeddings):
    # Loads your local FAISS database
    return FAISS.load_local("./vector_store", _embeddings, allow_dangerous_deserialization=True)

def main():
    st.set_page_config(page_title="GovTech Policy & Exam Navigator", layout="wide")
    st.title("GovTech Policy & Exam Navigator")

    with st.sidebar:
        st.subheader("API Key")
        api_key = st.text_input("Google Gemini API Key", type="password")
        st.markdown("Knowledge base running completely offline.")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Ask me about policies or exam details."}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a policy or exam question")
    if user_input:
        if not api_key:
            st.warning("Please enter your Google Gemini API key in the sidebar.")
            return

        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and thinking..."):
                try:
                    # 1. Retrieve the exact paragraphs from your PDFs
                    embeddings = load_embeddings()
                    vector_store = load_vector_store(embeddings)
                    docs = vector_store.similarity_search(user_input, k=3)
                    
                    # 2. Format the context with page number citations
                    context = "\n\n".join([f"Source: {d.metadata.get('source', 'Unknown')} | Page {d.metadata.get('page', '?')}\n{d.page_content}" for d in docs])
                    
                    # 3. Build the strict GovTech prompt
                    prompt = f"You are a professional GovTech advisor. Answer based ONLY on the context below. Include the Source and Page number in your answer.\n\nContext:\n{context}\n\nQuestion: {user_input}\n\nAnswer:"
                    
                    # 4. Connect to the most stable Gemini model
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-2.5-flash", 
                        google_api_key=api_key, 
                        temperature=0.1
                    )
                    
                    # 5. Get the response and display it
                    response = llm.invoke(prompt)
                    answer_text = response.content
                    
                    st.markdown(answer_text)
                    st.session_state.messages.append({"role": "assistant", "content": answer_text})
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()