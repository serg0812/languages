import streamlit as st
import psycopg2
from werkzeug.security import generate_password_hash, check_password_hash



# Database connection details
DB_NAME = st.secrets["DB_NAME"]
DB_USER = st.secrets["DB_USER"]
DB_PASS = st.secrets["DB_PASS"]
DB_HOST = st.secrets["DB_HOST"]
DB_PORT = st.secrets["DB_PORT"]

def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            host=DB_HOST,
            port=DB_PORT,
            sslmode='require' 
        )
        return conn
    except psycopg2.Error as e:
        st.error(f"Error connecting to database: {e}")
        return None

def create_user_table():
    conn = get_db_connection()
    if conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                name VARCHAR(50),
                surname VARCHAR(50),
                email VARCHAR(100) UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        cur.close()
        conn.close()

create_user_table()

def sign_up(name, surname, email, password):
    password_hash = generate_password_hash(password)
    conn = get_db_connection()
    if conn:
        cur = conn.cursor()
        try:
            cur.execute("""
                INSERT INTO users (name, surname, email, password_hash)
                VALUES (%s, %s, %s, %s)
            """, (name, surname, email, password_hash))
            conn.commit()
        except psycopg2.Error as e:
            st.error(f"Error: {e}")
        finally:
            cur.close()
            conn.close()

def sign_in(email, password):
    conn = get_db_connection()
    if conn:
        cur = conn.cursor()
        cur.execute("SELECT password_hash FROM users WHERE email = %s", (email,))
        user = cur.fetchone()
        cur.close()
        conn.close()
        if user and check_password_hash(user[0], password):
            return True
        else:
            return False
    return False
