import sqlite3

def get_connection():
    return sqlite3.connect("app.db")

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            status TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()

def create_dataset():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO datasets (status) VALUES (?)",
        ("Not Loaded",)
    )

    conn.commit()
    dataset_id = cursor.lastrowid
    conn.close()

    return dataset_id

def get_all_datasets():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM datasets"
    )

    conn.commit()
    result = cursor.fetchall()
    conn.close()

    return result