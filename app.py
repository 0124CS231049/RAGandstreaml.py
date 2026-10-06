
import os
import tempfile
from langchain_community.vectorstores import FAISS
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Streamlit page configuration
st.set_page_config(page_title="PDF RAG Chatbot",layout="wide")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []


st.title("PDF RAG Chatbot")
st.write("Upload a PDF file and ask questions about its content.")

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY not found in .env file.")
    st.stop()

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file is not None:
    os.makedirs("data", exist_ok=True)
    
    pdf_path = os.path.join("data", uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        st.success("PDF uploaded successfully!")
        
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        st.write(f"pages loaded: {len(documents)}")
        
        # Split text
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        st.write(f"Text chunks created: {len(chunks)}")
        
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # vector database
        vectorstore = FAISS.from_documents(chunks, embeddings)
        st.success("PDF processed successfully!")
        # Retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        
        #LLM
        llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0.2,
        )
        # Question
        question = st.text_input("Ask a question about the PDF:")
        if question:
            retrieved_docs = retriever.invoke(question)
            
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            prompt = f"""You are a helpful PDF question answering AI assistant.
            Answer the user's question using ONLY the information available in the context.
            If the answer is not available in the context, say: " I could not find the information in the PDF."
            
            Context: {context}
            Question: {question}
            Answer:"""
            with st.chat_message("assistant"):
                with st.spinner("Generating answer..."):
                    response = llm.invoke(prompt)
                    answer = response.content
                    st.markdown(answer)
                    
                    #Sources
                    st.markdown("### Sources")
                    shown_pages = set()
                    for doc in retrieved_docs:
                        page = doc.metadata.get("page")
                        if page is not None:
                            page_number = page + 1
                            if page_number not in shown_pages:
                                
                                st.write(f"-Page {page_number}")
                                shown_pages.add(page_number)
                                
                                # Save assistant
                            
                                st.session_state.messages.append(
                                    {
                                        "role":"assistant",
                                        "content":answer
                                    }
                                )
                            else:
                                st.info("Upload a PDF to start chatting.")
                            
                        

    
   
            
            