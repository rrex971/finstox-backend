import json
import nsepython as nse
import pandas as pd
from fastapi import FastAPI, Depends
from typing import Annotated
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import requests
import base64
import os
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from datetime import datetime, timedelta
import csv
from util.dbsetup import load_db
from util.auth import *

import asyncio

cur, conn = load_db()
nse_symbols = nse.nse_eq_symbols()

app = FastAPI(title="Finstox API")

async def run_nse_in_executor(func, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, func, *args)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/logos", StaticFiles(directory="logos"), name="logos")

@app.get("/")
async def root():
    return {"message": "Internal API for Finstox"}

@app.get("/isMarketOpen")
async def isMarketOpen():
    status_result = await run_nse_in_executor(nse.nse_marketStatus)
    return status_result["marketState"][0]["marketStatus"]!="Closed";

# Search endpoint
@app.get("/getSearchSuggestions")
async def getSearchSuggestions(query: str):
    suggestions = []
    c=0
    q_lower = query.lower()
    for symbol in nse_symbols:
        if c == 3: break
        symbol_lower = symbol.lower()
        if symbol_lower.startswith(q_lower) or q_lower in symbol_lower:
            c+=1
            stock = await run_nse_in_executor(nse.nse_eq, symbol)
            if stock and 'info' in stock and 'priceInfo' in stock:
                suggestions.append({
                    "symbol": symbol,
                    "name" : stock["info"]["companyName"],
                    "ltp" : stock["priceInfo"]["lastPrice"]
                    })
            else:
                 print(f"Warning: Could not fetch data for {symbol} in getSearchSuggestions")
    return {"suggestions": suggestions}
    

@app.get("/getSearchSuggestionsFull")
async def getSearchSuggestionsFull(query: str):
    if len(query) < 3:
        return {"detail": "Query too short"}, 400
    suggestions = []
    q_lower = query.lower()
    match_count = 0
    for symbol in nse_symbols:
        if match_count >= 20: break
        symbol_lower = symbol.lower()
        if symbol_lower.startswith(q_lower) or q_lower in symbol_lower:
            match_count += 1
            stock = await run_nse_in_executor(nse.nse_eq, symbol)
            if stock and 'info' in stock and 'priceInfo' in stock:
                try:
                    suggestions.append({
                        "name": stock["info"]["companyName"],
                        "symbol": symbol,
                        "price": stock["priceInfo"]["lastPrice"],
                        "onedaychange": round(abs(float(stock["priceInfo"]["change"])), 2),
                        "onedaychangepercent": round(abs(float(stock["priceInfo"]["pChange"])), 2),
                        "positive": float(stock["priceInfo"]["pChange"]) > 0
                    })
                except (KeyError, ValueError, TypeError) as e:
                    print(f"Error processing data for {symbol} in getSearchSuggestionsFull: {e}")
            else:
                print(f"Warning: Could not fetch data for {symbol} in getSearchSuggestionsFull")

    return {"suggestions": suggestions}

@app.get("/getTopGainers")
async def getTopGainers():
    # Run blocking call in executor
    top_gainers_df = await run_nse_in_executor(nse.nse_get_top_gainers)
    # Convert DataFrame to dict after getting the result
    top_gainers = top_gainers_df.to_dict(orient="records")
    gainers = []
    for i in top_gainers:
        try: # Add try-except for safety when accessing dict keys
            gainers.append({
                "name": i["meta"]["companyName"],
                "symbol": i["symbol"],
                "price": i["lastPrice"],
                "onedaychange": round(abs(float(i["change"])), 2),
                "onedaychangepercent": round(abs(float(i["pChange"])), 2),
                "positive": float(i["pChange"]) > 0,
                "logo": None
            })
        except (KeyError, ValueError, TypeError) as e:
             print(f"Error processing gainer item: {e}, item: {i}")
    return gainers

@app.get("/getTopLosers")
async def getTopLosers():
    # Run blocking call in executor
    top_losers_df = await run_nse_in_executor(nse.nse_get_top_losers)
    # Convert DataFrame to dict after getting the result
    top_losers = top_losers_df.to_dict(orient="records")
    losers = []
    for i in top_losers:
         try: # Add try-except for safety when accessing dict keys
            losers.append({
                "name": i["meta"]["companyName"],
                "symbol": i["symbol"],
                "price": i["lastPrice"],
                "onedaychange": round(abs(float(i["change"])), 2),
                "onedaychangepercent": round(abs(float(i["pChange"])), 2),
                "positive": float(i["pChange"]) > 0,
                "logo": None
            })
         except (KeyError, ValueError, TypeError) as e:
             print(f"Error processing loser item: {e}, item: {i}")

    return losers

company_websites = {}
try:
    with open("company_websites.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                company_websites[row[0]] = row[1]
except FileNotFoundError:
    print("Warning: company_websites.csv not found.")

@app.get("/getStock")
async def getStock(symbol: str):
    stock = await run_nse_in_executor(nse.nse_eq, symbol)

    if not stock or 'info' not in stock or 'priceInfo' not in stock:
         return JSONResponse(status_code=404, content={"detail": f"Could not retrieve data for symbol {symbol}"})

    logo = None
    website = company_websites.get(symbol)

    try:
        return {
            "name": stock["info"]["companyName"],
            "symbol": symbol,
            "price": stock["priceInfo"]["lastPrice"],
            "onedaychange": round(abs(float(stock["priceInfo"]["change"])), 2),
            "onedaychangepercent": round(abs(float(stock["priceInfo"]["pChange"])), 2),
            "positive": float(stock["priceInfo"]["pChange"]) > 0,
            "website": website,
            "industry": stock.get("industryInfo", {}).get("industry", "N/A"), # Safe access
        }
    except (KeyError, ValueError, TypeError) as e:
        print(f"Error processing stock data for {symbol}: {e}")
        return JSONResponse(status_code=500, content={"detail": f"Error processing data for symbol {symbol}"})

@app.get("/getHoldings")
async def getHoldings(username: str):
    holdings = cur.execute(f"""
        SELECT h.symbol, h.quantity, h.price
        FROM holdings h
        INNER JOIN users u
        ON h.userid = u.userid
        WHERE u.username = ?
    """, (username,)).fetchall()

    holdingsData = {"total":
        {
            "totalInvested": 0,
            "totalCurrent": 0,
            "totalChange": 0,
            "totalChangePercent": 0,
            "positive": False
        }, "holdings": []}

    total_invested = 0
    total_current = 0

    for holding in holdings:
        symbol, quantity, buy_price = holding[0], holding[1], holding[2]
        stock = await run_nse_in_executor(nse.nse_eq, symbol)

        if stock and 'info' in stock and 'priceInfo' in stock:
            try:
                last_price = float(stock["priceInfo"]["lastPrice"])
                totalbuy = buy_price * quantity
                totalcurrent = last_price * quantity

                total_invested += totalbuy
                total_current += totalcurrent

                change = totalcurrent - totalbuy

                holdingsData["holdings"].append({
                "name": stock["info"]["companyName"],
                "symbol": symbol,
                "price": last_price,
                "quantity": quantity,
                "buyPrice": buy_price,
                "currentPrice": last_price,
                "totalBuyPrice": totalbuy,
                "totalCurrentPrice": totalcurrent,
                "change": round(abs(change), 2),
                "changePercent": abs(round((change / totalbuy * 100), 2)) if totalbuy != 0 else 0,
                "positive": change > 0,
                })
            except (KeyError, ValueError, TypeError) as e:
                 print(f"Error processing holding {symbol} for {username}: {e}")
        else:
             print(f"Warning: Could not fetch data for holding {symbol} for user {username}")


    holdingsData["total"]["totalInvested"] = round(total_invested, 2)
    holdingsData["total"]["totalCurrent"] = round(total_current, 2)

    if holdingsData["total"]["totalInvested"] != 0:
        total_change_calc = holdingsData["total"]["totalCurrent"] - holdingsData["total"]["totalInvested"]
        holdingsData["total"]["totalChange"] = round(abs(total_change_calc), 2)
        holdingsData["total"]["totalChangePercent"] = abs(round((total_change_calc / holdingsData["total"]["totalInvested"] * 100), 2))
        holdingsData["total"]["positive"] = total_change_calc > 0

    return holdingsData

## login stuff

@app.get("/checkUsername")
async def checkUsername(query: str):
    cur.execute("SELECT username FROM users WHERE username = ?", (query.strip(),))
    if cur.rowcount > 0:
        return JSONResponse(status_code=400, content={"detail": "Username already taken"})
    else:
        return JSONResponse(status_code=200, content={"detail": "Username available"})

@app.post("/register")
async def register(data: dict):    
    
    password = data["password"]
    username = data["uname"]
    email = data["email"]
    cur.execute("SELECT username FROM users WHERE username = ?", (username,))
    if cur.rowcount > 0:
        return JSONResponse(status_code=400, content={"detail": "Username already taken"})
    hashed_password = get_password_hash(password)
    print(hashed_password)
    cur.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", (username, email, hashed_password))
    uid = cur.lastrowid
    cur.execute("INSERT INTO wallet (userid) VALUES (?)", (uid,))
    conn.commit()
    return JSONResponse(status_code=201, content={"detail": "Account registered successfully"})


@app.post("/login")
async def login(data: OAuth2PasswordRequestForm = Depends()):
    cur.execute("SELECT userid, password FROM users WHERE username = ?", (data.username,))
    user = cur.fetchall()
    if cur.rowcount==0 or not verify_password(data.password, user[0][1]):
        return JSONResponse(status_code=401, content={"detail": "Incorrect username or password"})
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user[0]}, expires_delta=access_token_expires
    )
    

    unique_key = base64.urlsafe_b64encode(os.urandom(24)).decode('utf-8')
    return {"access_token": access_token, "token_type": "bearer", "category" : "user"}

## wallet endpoints
@app.get("/getWallet")
async def getWallet(username: str):
    cur.execute("SELECT userid FROM users WHERE username = ?", (username,))
    user_result = cur.fetchone()
    if not user_result:
        return JSONResponse(status_code=404, content={"detail": "User not found"})
    user_id = user_result[0]
    cur.execute("SELECT balance FROM wallet WHERE userid = ?", (user_id,))
    balance = cur.fetchall()[0]
    return {"balance": balance}

@app.post("/deposit")
async def deposit(data: dict):
    username = data["username"]
    amount = data["amount"]

    if amount <= 0:
        return JSONResponse(status_code=400, content={"detail": "Amount must be greater than zero"})

    cur.execute("SELECT userid FROM users WHERE username = ?", (username,))
    user = cur.fetchone()
    if not user:
        return JSONResponse(status_code=404, content={"detail": "User not found"})

    user_id = user[0]
    cur.execute("UPDATE wallet SET balance = balance + ? WHERE userid = ?", (amount, user_id))
    conn.commit()
    return {"detail": "Deposit successful"}

@app.post("/withdraw")
async def withdraw(data: dict):
    username = data["username"]
    amount = data["amount"]
    if amount <= 0:
        return JSONResponse(status_code=400, content={"detail": "Amount must be greater than zero"})

    cur.execute("SELECT userid FROM users WHERE username = ?", (username,))
    user = cur.fetchone()
    if not user:
        return JSONResponse(status_code=404, content={"detail": "User not found"})

    user_id = user[0]

    cur.execute("SELECT balance FROM wallet WHERE userid = ?", (user_id,))
    balance = cur.fetchone()[0]
    if balance < amount:
        return JSONResponse(status_code=400, content={"detail": "Insufficient funds"})

    cur.execute("UPDATE wallet SET balance = balance - ? WHERE userid = ?", (amount, user_id))
    conn.commit()
    return {"detail": "Withdrawal successful"}


@app.get("/getQtyOwned")
async def getQtyOwned(username: str, symbol: str):

    try:
        cur.execute("""
            SELECT h.quantity
            FROM holdings h
            INNER JOIN users u ON h.userid = u.userid
            WHERE u.username = ? AND h.symbol = ?
        """, (username, symbol))

        result = cur.fetchone()

        if result:
            quantity = result[0]
        else:
            quantity = 0

        return {"quantity": quantity}

    except Exception as e:
        print(f"Error in getQtyOwned for user {username}, symbol {symbol}: {e}")
        return JSONResponse(status_code=500, content={"detail": "Internal server error fetching quantity"})