import streamlit as st
import requests
import json

# 设置页面标题和图标
st.set_page_config(page_title="Llama Chat", page_icon="🦙")

# 添加标题和描述
st.title("Llama Chat")
st.write("与 Llama 模型进行交互，获取实时响应。")

# 用户输入框
user_input = st.text_area("输入你的问题:", "9.9和9.11哪个大")

# 发送请求按钮
if st.button("发送"):
    with st.spinner("正在处理..."):
        def get_streamed_data():
            url = "https://llama3.bnnd.eu.org/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
            data = {
                "model": "llama-3.1-405b",
                "stream": True,
                "messages": [
                    {"role": "system", "content": "用中文回答"},
                    {"role": "user", "content": user_input}
                ]
            }

            # 使用 stream=True 发送请求以处理流响应
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
                                    # 实时更新响应内容
                                    st.markdown(response_text)
                            except json.JSONDecodeError:
                                continue
                return response_text

        # 调用函数并获取最终响应内容
        final_response = get_streamed_data()
        st.success("处理完成!")
