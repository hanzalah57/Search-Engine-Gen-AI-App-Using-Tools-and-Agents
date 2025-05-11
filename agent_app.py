# Importing Libraries
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

# Load environment variables
load_dotenv()

# Initialize Arxiv and Wikipedia Tools
# Create an arxiv wrapper to define how many results to fetch and how much content from each
arxiv_wrapper = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=300)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)  # Tool to query Arxiv using the wrapper

# Create a wikipedia wrapper with similar settings
wiki_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=300)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)  # Tool to query Wikipedia using wrapper

# Create a DuckDuckGo search tool for general purpose web search
search = DuckDuckGoSearchRun(name="Search")  # Name given to the search tool

# Streamlit UI Setup
st.set_page_config(layout="wide")
st.title("🔎 Search Engine (GEN AI APP) using Tools and Agents")
st.sidebar.header("🪛Configuration")
st.sidebar.write(
    " - Enter your Groq API Key \n "
    " - Ask or Search anything"
)

st.sidebar.header("🔑 Input Groq API Key")
api_key = st.sidebar.text_input("Please Enter your Groq API key:", type="password")  # API Key input

if not api_key:
    st.warning(" 🔑 Please enter your Groq API Key in the sidebar to continue. ")
    st.stop()

# A placeholder to show success/failure message
if api_key:
    # Replace this block with actual API key validation if possible
    try:
        # Dummy validation logic; replace with real Groq API call to verify
        if api_key.startswith("gsk_") and len(api_key) > 10:
            st.sidebar.success("✅ API Key authentication successful!")
        else:
            st.sidebar.error("❌ Invalid API Key. Please try again.")
    except Exception as e:
        st.sidebar.error(f"Error validating API key: {e}")

# Session state for chat messages
# If messages doesn't exist in session state, initialize it with a greeting message
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hi, I am a chatbot who can search the web. How can I help you?"
        }
    ]

# Loop through all past messages and display them in the chat window.
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])  # Display each message based on the sender role.

# Chat Input and Preprocessing

# When user inputs a new message in the chat
if prompt := st.chat_input(placeholder="What is machine learning?"):
    # Append user's message to session state message list
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Show the user's message in the chat interface
    st.chat_message("user").write(prompt)

    # Initialized ChatGroq LLM using provided API
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="Llama3-8b-8192",
        streaming=True
    )

    tools = [search, arxiv, wiki] 

    # We will create an agent that uses ZERO_SHOT_REACT_DESCRIPTION
    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handling_parsing_error=True
    )

    # Get and display the response
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        # Get the context from the session state messages
        context = "\n".join([msg['content'] for msg in st.session_state.messages])

        # Run the search agent with the updated context
        response = search_agent.run(context, callbacks=[st_cb])

        # Add the assistant's response to session state
        st.session_state.messages.append({
            'role': 'assistant',
            'content': response
        })

        # Display the assistant's response in the chat UI
        st.write(response)
