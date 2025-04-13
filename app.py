import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ---------------- Gemini Model Wrapper ----------------
class GeminiModelWrapper:
    def __init__(self, model_name="models/gemini-1.5-pro-latest", temperature=0.3):
        self.model = genai.GenerativeModel(model_name)
        self.temperature = temperature

    def generate_answer(self, context, question):
        prompt = f"""
        Answer the question as detailed as possible from the provided context.
        If the answer is not in the context, say: "Answer is not available in the context."

        Context: {context}

        Question: {question}

        Answer:
        """
        response = self.model.generate_content(prompt)
        return response.text

# ---------------- PDF Processing ----------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# ---------------- User Input ----------------
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    context = "\n\n".join([doc.page_content for doc in docs])

    model = GeminiModelWrapper(model_name="models/gemini-1.5-pro-latest")
    response = model.generate_answer(context, user_question)

    st.write("Reply:", response)

# ---------------- Streamlit App ----------------
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini 💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True
        )
        if st.button("Submit & Process") and pdf_docs:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
