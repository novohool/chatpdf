import streamlit as st
import requests
import json
import os
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ChatMessageHistory

def get_current_session():
    return st.session_state.session_id

def get_new_session(session_id='default_chat'):
    st.session_state.session_id = session_id

def get_session_history(session_id: str) -> ChatMessageHistory:
    session_id = get_current_session() if not session_id else session_id
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

def start_new_chat():
    new_session_id = f'chat_{len(st.session_state.store) + 1}'
    get_new_session(new_session_id)
    st.session_state.query = ""
    st.session_state.store[new_session_id] = ChatMessageHistory()

def initialize_session_state():
    if 'store' not in st.session_state:
        st.session_state.store = {}
    if 'session_id' not in st.session_state:
        get_new_session()
    if 'query' not in st.session_state:
        st.session_state.query = ""

class LlamaChat:
    def __init__(self):
        st.set_page_config(page_title="Llama Chat", page_icon="🦙")
        st.title("Llama Chat")
        st.write("与 Llama 模型进行交互，获取实时响应。")
        
        initialize_session_state()

        if 'button_key' not in st.session_state:
            st.session_state.button_key = 0
        if 'vectorstore' not in st.session_state:
            st.session_state.vectorstore = None

        self.add_custom_css()

    def add_custom_css(self):
        custom_css = """
        <style>
        .stButton button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            cursor: pointer;
            transition: transform 0.2s ease, background-color 0.3s ease;
            border-radius: 0;
        }
        .stButton button:hover {
            background-color: #45a049;
            transform: scale(1.05);
        }
        .stTextInput input, .stTextArea textarea {
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 0;
            transition: border-color 0.3s ease;
        }
        .stTextInput input:focus, .stTextArea textarea:focus {
            border-color: #4CAF50;
            outline: none;
        }
        .stMarkdown p {
            background-color: #f7f7f7;
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            border: none;
        }
        .stForm {
            padding: 20px;
            background-color: white;
            box-shadow: none;
            border-radius: 0;
        }
        .stSpinner {
            font-size: 16px;
            color: #4CAF50;
        }
        </style>
        """
        st.markdown(custom_css, unsafe_allow_html=True)

    @st.cache_resource(ttl="1h", show_spinner='Processing File(s).')
    def get_docs_from_files(self, files):
        documents = []
        for file in files:
            filepath = file.name
            with open(filepath, "wb") as f:
                f.write(file.getvalue())
    
            try:
                docs = PyPDFLoader(filepath).load()
                documents.extend(docs)
            except UnicodeDecodeError:
                st.error(f"文件 {filepath} 包含非 UTF-8 字符，无法读取。")
    
            if os.path.exists(filepath):
                os.remove(filepath)
    
        return documents
        
    @st.cache_resource(ttl="1h", show_spinner='Processing File(s)..')
    def get_vectorstore_from_files(self, files, HF_Embed_Model):
        pdf_docs = self.get_docs_from_files(files)             
        split_docs = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150).split_documents(pdf_docs)
        embeddings = HuggingFaceEmbeddings(model_name=HF_Embed_Model)
        vectorstore = FAISS.from_documents(split_docs, embeddings)
    
        return vectorstore

    def get_streamed_data(self, user_input):
        url = "https://llama3.bnnd.eu.org/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        history = get_session_history(get_current_session()).messages
        data = {
            "model": "llama-3.1-405b",
            "stream": True,
            "messages": [
                {"role": "system", "content": "用中文回答"}
            ] + history + [{"role": "user", "content": user_input}]
        }

        try:
            with requests.post(url, headers=headers, json=data, stream=True) as response:
                response_text = ""
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8', errors='ignore')
                        if decoded_line.startswith("data: "):
                            json_data = decoded_line[6:]
                            if json_data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(json_data)
                                if "choices" in chunk and chunk["choices"]:
                                    content = chunk["choices"][0]["delta"].get("content", "")
                                    response_text += content
                            except json.JSONDecodeError:
                                continue
                return response_text
        except requests.RequestException as e:
            st.error(f"请求失败: {e}")
            return None

    def display_history(self):
        st.write("历史消息:")
        history = get_session_history(get_current_session()).messages
        for message in history:
            if message.role == "user":
                st.markdown(f"**用户：** {message.content}")
            else:
                st.markdown(f"**助手：** {message.content}")

    def rag_chain(self, user_input):
        retriever = st.session_state.vectorstore.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=None, retriever=retriever, chain_type="stuff")
        response = qa_chain.run(user_input)
        return response

    def main_fragment(self):
        with st.form(key=f"form_{st.session_state.button_key}", clear_on_submit=True):
            user_input = st.text_area("输入你的问题:", "", key=f"input_{st.session_state.button_key}")
            submit_button = st.form_submit_button(label="发送")

            if submit_button:
                with st.spinner("正在处理..."):
                    final_response = self.rag_chain(user_input)
                    if final_response:
                        history = get_session_history(get_current_session())
                        history.add_user_message(user_input)
                        history.add_ai_message(final_response)
                        st.markdown(final_response)
                        st.success("处理完成!")
                        st.session_state.button_key += 1
                    else:
                        st.error("处理失败，请重试。")

        self.display_history()

    def main(self):
        files = st.file_uploader('Choose PDF file(s)', type=['pdf'], accept_multiple_files=True)
        if files:
            HF_Embed_Model = "sentence-transformers/all-MiniLM-L6-v2"
            vectorstore = self.get_vectorstore_from_files(files=files, HF_Embed_Model=HF_Embed_Model)
            st.session_state.vectorstore = vectorstore  # 存储 vectorstore 到 session_state
            st.success("向量存储已生成！")

        self.main_fragment()

if __name__ == "__main__":
    chat = LlamaChat()
    chat.main()
