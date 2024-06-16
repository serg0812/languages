# main.py
import streamlit as st
from auth import sign_up, sign_in
import os
import psycopg2
from psycopg2 import sql
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
import pandas as pd
import numpy as np
import time
import uuid

# Set page configuration
st.set_page_config(page_title="Welcome to our language school", initial_sidebar_state="auto", menu_items=None)

DB_NAME = st.secrets["DB_NAME"]
DB_USER = st.secrets["DB_USER"]
DB_PASS = st.secrets["DB_PASS"]
DB_HOST = st.secrets["DB_HOST"]
DB_PORT = st.secrets["DB_PORT"]

# Function to clear session state
def clear_session_state():
    for key in st.session_state.keys():
        del st.session_state[key]

# User authentication
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

if not st.session_state['authenticated']:
    auth_action = st.sidebar.selectbox('Select Action', ['Sign In', 'Sign Up'])

    if auth_action == 'Sign Up':
        st.title('Sign Up')
        name = st.text_input('First Name')
        surname = st.text_input('Surname')
        email = st.text_input('Email')
        password = st.text_input('Password', type='password')
        if st.button('Sign Up'):
            sign_up(name, surname, email, password)
            st.success("You have successfully signed up! Please sign in.")

    if auth_action == 'Sign In':
        st.title('Sign In')
        email = st.text_input('Email')
        password = st.text_input('Password', type='password')
        if st.button('Sign In'):
            if sign_in(email, password):
                st.session_state['authenticated'] = True
                st.session_state['email'] = email
                st.session_state['session_id'] = str(uuid.uuid4())
                # Fetch user details from the database
                conn = psycopg2.connect(
                    dbname=DB_NAME,
                    user=DB_USER,
                    password=DB_PASS,
                    host=DB_HOST,
                    port=DB_PORT,
                    sslmode='require' 
                )
                cur = conn.cursor()
                cur.execute("SELECT name, surname FROM users WHERE email = %s", (email,))
                user = cur.fetchone()
                cur.close()
                conn.close()
                st.session_state['name'] = user[0]
                st.session_state['surname'] = user[1]
                st.rerun()
            else:
                st.error("Invalid email or password")
else:
    st.sidebar.button('Sign Out', on_click=clear_session_state)

    # The existing conversation code
    def create_table():
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            host=DB_HOST,
            port=DB_PORT,
            sslmode='require' 
        )
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(100),
                name VARCHAR(50),
                surname VARCHAR(50),
                session_id VARCHAR(36),
                AI BOOLEAN,
                text TEXT,
                timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        cur.close()
        conn.close()

    create_table()

    def get_db_connection():
        if 'db_connection' not in st.session_state or st.session_state['db_connection'].closed:
            st.session_state['db_connection'] = psycopg2.connect(
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASS,
                host=DB_HOST,
                port=DB_PORT,
                sslmode='require' 
            )
        return st.session_state['db_connection']

    def close_db_connection():
        if 'db_connection' in st.session_state and not st.session_state['db_connection'].closed:
            st.session_state['db_connection'].close()

    def save_message(user_id, name, surname, session_id, AI, message):
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO chat_history (user_id, name, surname, session_id, AI, text)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (user_id, name, surname, session_id, AI, message))
        conn.commit()
        cur.close()

    # Define the inactivity timeout (5 minutes)
    INACTIVITY_TIMEOUT = 300

    @tool   
    def get_details(name: str, surname: str) -> str:
        """tool to get name and surname"""
        return f"now you are dealing with {name} {surname}"

    with open("instructions.txt") as file:
        instructions = file.read()
    text = """
    You are the best language teacher in the world. 
    You teach any language to the students from any origin.
    Your detailed instructions are here: 
    """ + f"{instructions}"

    llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=4000)
    tools = [get_details]
    llm_with_tools = llm.bind_tools(tools=tools)

    MEMORY_KEY = "chat_history"
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", f"{text}"),
            MessagesPlaceholder(variable_name=MEMORY_KEY),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
        st.session_state['last_interaction_time'] = time.time()

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
            "chat_history": lambda x: x["chat_history"],
        }
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    st.markdown("<h2 style='text-align: center; color: grey;'>📱 Welcome to our language school 📱</h2>", unsafe_allow_html=True)

    user_input = st.chat_input("")

    if user_input:
        st.session_state['chat_history'].append(HumanMessage(content=user_input))
        save_message(st.session_state['email'], st.session_state['name'], st.session_state['surname'], st.session_state['session_id'], False, user_input)
        st.session_state['last_interaction_time'] = time.time()

        result = agent_executor.invoke({
            "input": user_input,
            "chat_history": st.session_state['chat_history']
        })
        
        st.session_state['chat_history'].append(AIMessage(content=result["output"]))
        save_message(st.session_state['email'], st.session_state['name'], st.session_state['surname'], st.session_state['session_id'], True, result["output"])
        st.session_state['last_interaction_time'] = time.time()

    st.write("--------------------------------------")
    for message in st.session_state['chat_history']:
        if isinstance(message, HumanMessage):
            st.markdown(f"**You:** {message.content}")
        elif isinstance(message, AIMessage):
            st.markdown(f"**XYZ:** {message.content}")

    # Check for inactivity
    if time.time() - st.session_state['last_interaction_time'] > INACTIVITY_TIMEOUT:
        close_db_connection()

    # Close the database connection when the user closes the window
#    st.session_state.on_script_shutdown(close_db_connection)
