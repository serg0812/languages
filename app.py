import streamlit as st
from auth import sign_up, sign_in
from database import create_table, get_db_connection, close_db_connection, save_message
from chat import load_last_conversation, update_level
import os
import psycopg2
import time
import uuid
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents import AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_community.callbacks import get_openai_callback
from langchain_core.tools import StructuredTool

# Set page configuration
st.set_page_config(page_title="Welcome to our language school", initial_sidebar_state="auto", menu_items=None)

DB_NAME = st.secrets["DB_NAME"]
DB_USER = st.secrets["DB_USER"]
DB_PASS = st.secrets["DB_PASS"]
DB_HOST = st.secrets["DB_HOST"]
DB_PORT = st.secrets["DB_PORT"]

# Define the inactivity timeout (5 minutes)
INACTIVITY_TIMEOUT = 300

# Function to clear session state
def clear_session_state():
    for key in st.session_state.keys():
        del st.session_state[key]

# Initialize database table
create_table()

# Define the tool function
def level_tool(level: int, session_id: str) -> int:
    return update_level(level, session_id)

# Wrap the tool function using StructuredTool
level_tool_def = StructuredTool.from_function(
    level_tool,
    name="level",
    description="Assign level of knowledge to the student and update records in the database"
)

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

                chat_history, level, lesson, method = load_last_conversation(st.session_state['email'])
                if chat_history:
                    st.session_state['chat_history'] = chat_history
                    st.session_state['level'] = level
                    st.session_state['lesson'] = lesson
                    st.session_state['method'] = method
                else:
                    st.session_state['chat_history'] = []
                    st.session_state['level'] = None
                    st.session_state['lesson'] = 1
                    st.session_state['method'] = None

                # Always start a new session ID
                st.session_state['session_id'] = str(uuid.uuid4())

                st.rerun()
            else:
                st.error("Invalid email or password")
else:
    st.sidebar.button('Sign Out', on_click=clear_session_state)

    session_id = st.session_state['session_id']
    name = st.session_state['name']

    with open("level_0.txt") as file:
        level_0_instructions = file.read()

    with open("instructions.txt") as file:
        instructions = file.read()

    # Determine the appropriate system prompt text and store it in session state
    if 'system_prompt' not in st.session_state:
        if st.session_state.get('chat_history') and st.session_state['level']:
            st.session_state['system_prompt'] = f"""
            You are the teacher in the best language school in the world. 
            You specialize in teaching level {st.session_state['level']} students.
            You can teach any language to the student with any native language.
            The student name is {name}.
            Your detailed instructions are here: 
            """ + f"{level_0_instructions}"
        else:
            st.session_state['system_prompt'] = f"""
            You are a director of the best language school in the world. 
            Your instructors can teach any language to the students from any origin.
            This session has this id {session_id}.
            The student name is {name}.
            Your detailed instructions are here: 
            """ + f"{instructions}"

    text = st.session_state['system_prompt']

    llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=4000)
    tools = [level_tool_def]
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
    else:
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

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, stream_runnable=False)

    st.markdown("<h2 style='text-align: center; color: grey;'>📱 Welcome to our language school 📱</h2>", unsafe_allow_html=True)

    user_input = st.chat_input("")

    if user_input:
        st.session_state['chat_history'].append(HumanMessage(content=user_input))
        save_message(st.session_state['email'], st.session_state['name'], st.session_state['surname'], st.session_state['session_id'], False, user_input, st.session_state.get('level'), st.session_state.get('lesson', 1), st.session_state.get('method'))
        st.session_state['last_interaction_time'] = time.time()

        # Tokens counting and executing chain
        with get_openai_callback() as cb:
            result = agent_executor.invoke({
                "input": user_input,
                "chat_history": st.session_state['chat_history']
            })
            print(cb.total_tokens)
        st.session_state['chat_history'].append(AIMessage(content=result["output"]))
        save_message(st.session_state['email'], st.session_state['name'], st.session_state['surname'], st.session_state['session_id'], True, result["output"], st.session_state.get('level'), st.session_state.get('lesson', 1), st.session_state.get('method'))
        st.session_state['last_interaction_time'] = time.time()

    st.write("--------------------------------------")
    for message in st.session_state['chat_history']:
        if isinstance(message, HumanMessage):
            st.markdown(f"**You:** {message.content}")
        elif isinstance(message, AIMessage):
            st.markdown(f"**XYZ:** {message.content}")

    # Check for inactivity
    if 'last_interaction_time' in st.session_state and time.time() - st.session_state['last_interaction_time'] > INACTIVITY_TIMEOUT:
        close_db_connection()
