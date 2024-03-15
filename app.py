import streamlit as st
from streamlit_extras.app_logo import add_logo
import time
from dotenv import load_dotenv
import pickle
import google.generativeai as GoogleGenerativeAI
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

add_logo("https://github.com/Youcef-bel/KetabAI/blob/main/Untitled%20design%20(4).png?raw=true")
# Sidebar contents
logo_url = "https://github.com/Youcef-bel/KetabAI/blob/main/Untitled%20design%20(4).png?raw=true"
with st.sidebar:
    st.sidebar.image(logo_url)
    st.title('KetabAI')
    st.markdown('''
    ## About
    An APP to help you discover your PDF without the need of reading All of it
 
    ''')
    
    st.write('Developed by [Youcef Belmokhtar](https://www.linkedin.com/in/youcefbelmokhtar/)')
 
load_dotenv()
 
def main():
    st.header("Chat with PDF 💬")
 
 
    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
 
    # st.write(pdf)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)
 
        # # embeddings
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')
        # st.write(chunks)
 
      
        embeddings = GooglePalmEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
           
 
        # embeddings = OpenAIEmbeddings()
        # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
 
        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")
        # st.write(query)
 
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            
                
 
            llm = GooglePalm(model="models/text-bison-001",temperature=0.4)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            with st.spinner('Wait for it...'):
                 time.sleep(3)
            st.write(response)
         
        
        st.markdown("---")
        # creating a button for Prediction
        sum="Summarize this PDF"
        if st.button('Summarize!'):
           docs = VectorStore.similarity_search(query=sum, k=3)
           llm = GooglePalm(model="models/text-bison-001",temperature=0.4)
           chain = load_qa_chain(llm=llm, chain_type="stuff")
           with get_openai_callback() as cb:
                response2 = chain.run(input_documents=docs, question=sum)
                print(cb)
           with st.spinner('Wait for it...'):
                time.sleep(3)
           st.write(response2)
 
if __name__ == '__main__':
    main()
