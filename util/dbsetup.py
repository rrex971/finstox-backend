import sqlite3

def load_db():
    global c, db
    db = sqlite3.connect("data.db", isolation_level=None)
    db.execute("PRAGMA journal_mode=WAL")
    c = db.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    userid INTEGER PRIMARY KEY AUTOINCREMENT, 
                    username TEXT NOT NULL UNIQUE,
                    email TEXT NOT NULL UNIQUE,
                    password TEXT NOT NULL
                )''')
    c.execute('''CREATE TABLE IF NOT EXISTS wallet (
                userid INTEGER PRIMARY KEY,
                balance REAL NOT NULL DEFAULT 0.0, 
                FOREIGN KEY(userid) REFERENCES users(userid)
                )''')
    c.execute('''CREATE TABLE IF NOT EXISTS orders (
                orderid INTEGER PRIMARY KEY AUTOINCREMENT,
                userid INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                type TEXT CHECK(type IN ('BUY', 'SELL')) NOT NULL,
                price REAL NOT NULL,
                FOREIGN KEY(userid) REFERENCES users(userid)
                )''')
    c.execute('''CREATE TABLE IF NOT EXISTS holdings (
                userid INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                FOREIGN KEY(userid) REFERENCES users(userid)
                )''')
    return c, db





