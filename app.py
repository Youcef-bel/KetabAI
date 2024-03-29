import streamlit as st
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

# Sidebar contents
logo_url = "https://github.com/Youcef-bel/KetabAI/blob/main/Untitled%20design%20(4).png?raw=true"
with st.sidebar:
    st.markdown('<div class="center"><img src="https://github.com/Youcef-bel/KetabAI/blob/main/Untitled%20design%20(4).png?raw=true" alt="Logo" width=150 height=150></div>', unsafe_allow_html=True)
    st.title('KetabAI')
    st.markdown("---")
    st.markdown('''
    ## About
    A PDF reading app designed to streamline your reading experience. With KetabAI, navigating through PDF documents has never been simpler. You can chat with your PDF file or summarize it with a single click!
 
    ''')
    st.markdown("---")
    st.write('Developed by [Youcef Belmokhtar](https://www.linkedin.com/in/youcefbelmokhtar/)')
 
load_dotenv()
 
def main():
    st.subheader("Chat with your PDF")
 
 
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
        st.subheader("PDF Summary")
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

        st.markdown("---")
        st.subheader("Similar Books")
        if st.button('Suggest!'):
            docs = VectorStore.similarity_search(query=sum, k=3)
            llm = GooglePalm(model="models/text-bison-001",temperature=0.4)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response2 = chain.run(input_documents=docs, question=sum)
            prompt = "Suggest similiar book' names with the same context:"+response2
            completion = GoogleGenerativeAI.generate_text(model="models/text-bison-001",prompt=prompt,temperature=0.7, max_output_tokens=800)
            print(completion.result)
            with st.spinner('Wait for it...'):
                time.sleep(3)
            st.write(completion.result)
 
 
if __name__ == '__main__':
    main()
