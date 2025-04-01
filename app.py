import os
import streamlit as st
from groq import Groq

st.set_page_config(
    page_title="Psychology AI Assistant",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4B0082;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #6A5ACD;
        margin-bottom: 1rem;
    }
    .response-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #6A5ACD;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>Psychology AI Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p>Powered by Groq's LLM API</p>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<h2 class='sub-header'>Model Settings</h2>", unsafe_allow_html=True)
    
    model = st.selectbox(
        "Select Model",
        ["llama-3.3-70b-versatile", "gemma2-9b-it", "mixtral-8x7b-32768"]
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Higher values make output more random, lower values more deterministic"
    )
    
    max_tokens = st.slider(
        "Max Completion Tokens",
        min_value=50,
        max_value=1000,
        value=100,
        step=50,
        help="Maximum number of tokens in the response"
    )
    
    top_p = st.slider(
        "Top P",
        min_value=0.1,
        max_value=1.0,
        value=1.0,
        step=0.1,
        help="Controls diversity via nucleus sampling"
    )
    
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        help="Enter your Groq API key"
    )
    
    st.markdown("---")
    st.markdown("Created with Streamlit and Groq API")

col1, col2 = st.columns([3, 3])

with col1:
    st.markdown("<h2 class='sub-header'>Input</h2>", unsafe_allow_html=True)
    
    system_role = st.text_area(
        "System Role",
        value="You are a psychology expert; please provide answers related exclusively to psychology.",
        height=100,
        help="Define the AI's role and behavior"
    )
    
    user_question = st.text_area(
        "Your Question",
        value="I feel very relaxed while working and drinking. What are the benefits and drawbacks of this habit?",
        height=150,
        help="Enter your question for the AI"
    )
    
    stop_sequence = st.text_input(
        "Stop Sequence (Optional)",
        value="",
        help="Text that signals the AI to stop generating content"
    )
    
    stream_option = st.checkbox("Stream Response", value=False)

submit_button = st.button("Get Response", type="primary")

with col2:
    st.markdown("<h2 class='sub-header'>Response</h2>", unsafe_allow_html=True)
    
    response_placeholder = st.empty()

if submit_button:
    if not api_key:
        st.error("Please enter your Groq API key in the sidebar")
    else:
        try:
            with st.spinner("Generating response..."):
                client = Groq(api_key=api_key)
                
                messages = [
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": user_question}
                ]
                
                stop_param = stop_sequence if stop_sequence else None
                
                if stream_option:
                    response_text = ""
                    response_container = response_placeholder.markdown(
                        "<div class='response-container'></div>", 
                        unsafe_allow_html=True
                    )
                    
                    for chunk in client.chat.completions.create(
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        max_completion_tokens=max_tokens,
                        top_p=top_p,
                        stop=stop_param,
                        stream=True
                    ):
                        content = chunk.choices[0].delta.content
                        if content is not None:
                            response_text += content
                            response_container.markdown(
                                f"<div class='response-container'>{response_text}</div>", 
                                unsafe_allow_html=True
                            )
                else:
                    chat_completion = client.chat.completions.create(
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        max_completion_tokens=max_tokens,
                        top_p=top_p,
                        stop=stop_param,
                        stream=False
                    )
                    
                    response_text = chat_completion.choices[0].message.content
                    response_placeholder.markdown(
                        f"<div class='response-container'>{response_text}</div>", 
                        unsafe_allow_html=True
                    )
                    
        except Exception as e:
            st.error(f"Error: {str(e)}")