import streamlit as st
#all agent imports
from langchain_openai import ChatOpenAI
from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents import AgentExecutor
from langchain.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

#for data manipulation
import pandas as pd
import numpy as np

@tool   
def get_details(name: str, surname: str) -> str: #, dob: int
    """
    Returns the details of the user based on name and surname
    """
    return f"now you are dealing with {name} {surname}"
    


#Agent block

#Instructions
with open("instructions.txt") as file:
    instructions=file.read()
text = """
You are the best language teacher in the world. 
You teach any language to the students from any origin.
Your detailed instructions are here: 
""" + f"{instructions}"

#model
llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=4000)
tools=[get_details]

llm_with_tools = llm.bind_tools(tools=tools)

#prompt
MEMORY_KEY = "chat_history"
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"{text}",
        ),
        MessagesPlaceholder(variable_name=MEMORY_KEY),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Ensure chat history is stored in session state
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

#agent
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

#Input part 

st.set_page_config(page_title="Welcome to our language school", initial_sidebar_state="auto", menu_items=None)
#openai.api_key = st.secrets.openai_key
st.markdown("<h2 style='text-align: center; color: grey;'>📱 Welcome to our language school 📱</h2>", unsafe_allow_html=True)
 
# Create a text area for user input
user_input = st.chat_input("")

# Create a button to trigger the agent
if user_input:
    # Add the user message to the chat history
    st.session_state['chat_history'].append(HumanMessage(content=user_input))

    # Run the agent with the chat history and user input
    result = agent_executor.invoke({
        "input": user_input,
        "chat_history": st.session_state['chat_history']
    })
    
    # Add the AI's response to the chat history
    st.session_state['chat_history'].append(AIMessage(content=result["output"]))
    

# Display the chat history
st.write("--------------------------------------")
for message in st.session_state['chat_history']:
    if isinstance(message, HumanMessage):
        st.markdown(f"**You:** {message.content}")
    elif isinstance(message, AIMessage):
        st.markdown(f"**XYZ:** {message.content}")

