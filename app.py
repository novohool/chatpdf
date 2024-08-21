import streamlit as st
import requests
import json

class LlamaChat:
    def __init__(self):
        # 设置页面标题和图标
        st.set_page_config(page_title="Llama Chat", page_icon="🦙")
        # 添加标题和描述
        st.title("Llama Chat")
        st.write("与 Llama 模型进行交互，获取实时响应。")
        # 初始化历史消息列表
        if 'history' not in st.session_state:
            st.session_state.history = []
        # 初始化按钮状态
        if 'button_key' not in st.session_state:
            st.session_state.button_key = 0

        # 添加自定义 CSS
        self.add_custom_css()

    def add_custom_css(self):
        custom_css = """
        <style>
        .stButton button {
            background-color: #4CAF50; /* 鲜绿色 */
            color: white;
            border-radius: 20px;
            padding: 10px 20px;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .stButton button:hover {
            background-color: #45a049; /* 深绿色 */
        }
        .stTextInput input, .stTextArea textarea {
            border-radius: 20px;
            padding: 10px;
            border: 1px solid #ccc;
            transition: border-color 0.3s ease;
        }
        .stTextInput input:focus, .stTextArea textarea:focus {
            border-color: #4CAF50; /* 鲜绿色 */
            outline: none;
        }
        .stMarkdown p {
            background-color: #e0f7fa; /* 蔚蓝色 */
            border-radius: 10px;
            padding: 10px;
            margin: 10px 0;
        }
        </style>
        """
        st.markdown(custom_css, unsafe_allow_html=True)

    def get_streamed_data(self, user_input):
        url = "https://llama3.bnnd.eu.org/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": "llama-3.1-405b",
            "stream": True,
            "messages": [
                {"role": "system", "content": "用中文回答"}
            ] + st.session_state.history + [{"role": "user", "content": user_input}]
        }

        try:
            with requests.post(url, headers=headers, json=data, stream=True) as response:
                response_text = ""
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
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
        for message in st.session_state.history:
            if message["role"] == "user":
                st.write(f"用户：{message['content']}")
            else:
                st.write(f"助手：{message['content']}")

    def main(self):
        # 用户输入框
        user_input = st.text_area("输入你的问题:", "9.9和9.11哪个大", key=f"input_{st.session_state.button_key}")

        if st.button("发送", key=f"button_{st.session_state.button_key}"):
            with st.spinner("正在处理..."):
                final_response = self.get_streamed_data(user_input)
                if final_response:
                    st.session_state.history.append({"role": "user", "content": user_input})
                    st.session_state.history.append({"role": "assistant", "content": final_response})
                    st.markdown(final_response)
                    st.success("处理完成!")
                st.session_state.button_key += 1
                st.experimental_rerun()

        self.display_history()

if __name__ == "__main__":
    chat = LlamaChat()
    chat.main()
