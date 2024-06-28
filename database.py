import psycopg2
import streamlit as st

DB_NAME = st.secrets["DB_NAME"]
DB_USER = st.secrets["DB_USER"]
DB_PASS = st.secrets["DB_PASS"]
DB_HOST = st.secrets["DB_HOST"]
DB_PORT = st.secrets["DB_PORT"]

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
            timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            level INT DEFAULT NULL,
            lesson INT DEFAULT 1,
            method VARCHAR(255) DEFAULT NULL
        )
    """)
    conn.commit()
    cur.close()
    conn.close()

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

def save_message(user_id, name, surname, session_id, AI, message, level=None, lesson=1, method=None):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO chat_history (user_id, name, surname, session_id, AI, text, level, lesson, method)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (user_id, name, surname, session_id, AI, message, level, lesson, method))
    conn.commit()
    cur.close()
