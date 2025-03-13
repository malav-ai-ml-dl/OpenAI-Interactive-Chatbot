import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Streamlit App Configuration
st.set_page_config(page_title="OpenAI Q&A Chatbot", page_icon="💬")

# Sidebar for API Key Input
st.sidebar.title("🔑 API Settings")
openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

st.sidebar.title("⚙️ Model Settings")
llm_model = st.sidebar.selectbox("Select OpenAI Model", ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"])
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

st.title("💬 OpenAI Q&A Chatbot")
st.write("Ask me anything, and I'll answer using OpenAI!")

# LangChain Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to user queries."),
    ("user", "Question: {question}")
])

# Function to Generate Response
def generate_response(question, api_key):
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar to continue.")
        return None
    
    try:
        llm = ChatOpenAI(api_key=api_key, model=llm_model, temperature=temperature, max_tokens=max_tokens)
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser
        answer = chain.invoke({'question': question})
        return answer
    except Exception as e:
        st.error(f"Error in generating response: {e}")
        return None

# User Input Section
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input, openai_api_key)
    if response:
        st.write(f"**Assistant:** {response}")
else:
    st.write("Please enter your question above to get a response.")
