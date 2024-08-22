import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

def main():
    # 设置页面配置
    st.set_page_config(page_title="Ask your PDF", page_icon="📄")
    st.header("Ask your PDF 💬")
    
    # 自定义CSS样式
    st.markdown("""
        <style>
        .human-message {
            text-align: right;
            background: linear-gradient(135deg, #a8e063, #56ab2f);
            padding: 14px;
            border-radius: 20px 20px 0 20px;
            margin-bottom: 12px;
            box-shadow: 0px 5px 10px rgba(0, 0, 0, 0.15);
            max-width: 65%;
            margin-left: auto;
            font-family: 'Roboto', sans-serif;
            font-size: 15px;
            color: #fff;
            animation: fade-slide-in 0.4s ease;
        }

        .ai-message {
            text-align: left;
            background: linear-gradient(135deg, #f0f0f0, #cccccc);
            padding: 14px;
            border-radius: 20px 20px 20px 0;
            margin-bottom: 12px;
            box-shadow: 0px 5px 10px rgba(0, 0, 0, 0.15);
            max-width: 65%;
            margin-right: auto;
            font-family: 'Roboto', sans-serif;
            font-size: 15px;
            color: #333;
            animation: fade-slide-in 0.4s ease;
        }

        @keyframes fade-slide-in {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        </style>
        """, unsafe_allow_html=True)
    
    # 上传PDF文件
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    if pdf is not None:
        with st.spinner('Extracting text from PDF...'):
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        
        # 分割文本为块
        with st.spinner('Splitting text into chunks...'):
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=600,
                chunk_overlap=20,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
        
        # 创建嵌入
        with st.spinner('Creating embeddings...'):
            embeddings = OllamaEmbeddings(model="quentinz/bge-large-zh-v1.5:f32")
            knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        # 提示模板
        template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        {context}
        Question: {question}
        Helpful Answer,请用中文回答:"""
        
        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question"],
            template=template,
        )
        
        # 用户输入
        user_question = st.text_input("Ask a question about your PDF:", key="input")
        if user_question:
            with st.spinner('Searching for answer...'):
                llm = Ollama(model="openllm/causallm:14b-dpo-alpha.Q5_K_M") 
                qa_chain = RetrievalQA.from_chain_type(
                    llm,
                    retriever=knowledge_base.as_retriever(),
                    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
                )
                response = qa_chain.invoke({"query": user_question})
                st.markdown(f"<div class='ai-message'>{response['result']}</div>", unsafe_allow_html=True)
    
if __name__ == '__main__':
    main()
