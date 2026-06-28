import sqlite3
conn = sqlite3.connect("audioforensics.db")
conn.execute("UPDATE users SET role='admin' WHERE username='AshleyNex'")
conn.commit()
conn.close()