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

cur, conn = load_db()
nse_symbols = nse.nse_eq_symbols()

app = FastAPI(title="Finstox API")


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
    return nse.nse_marketStatus()["marketState"][0]["marketStatus"]!="Closed";
# Search endpoint
@app.get("/getSearchSuggestions")
async def getSearchSuggestions(query: str):
    suggestions = []
    c=0
    for symbol in nse_symbols:
        if c == 3: break
        if symbol.lower().startswith(query.lower()) or query.lower() in symbol.lower():
            c+=1
            stock = nse.nse_eq(symbol)
            suggestions.append({
                "symbol": symbol,
                "name" : stock["info"]["companyName"],
                "ltp" : stock["priceInfo"]["lastPrice"]
                })
    return {"suggestions": suggestions}
    
@app.get("/getSearchSuggestionsFull")
async def getSearchSuggestionsFull(query: str):
    if len(query) < 3:
        return {"detail": "Query too short"}, 400
    suggestions = []
    for symbol in nse_symbols:
        if symbol.lower().startswith(query.lower()) or query.lower() in symbol.lower():
            stock = nse.nse_eq(symbol)
            logo = None
            suggestions.append({
                "name": stock["info"]["companyName"],
                "symbol": symbol,
                "price": stock["priceInfo"]["lastPrice"],
                "onedaychange": round(abs(float(stock["priceInfo"]["change"])), 2),
                "onedaychangepercent": round(abs(float(stock["priceInfo"]["pChange"])), 2),
                "positive": float(stock["priceInfo"]["pChange"]) > 0,
                "logo": logo        
            })
    return {"suggestions": suggestions}
    
@app.get("/getTopGainers")
async def getTopGainers():
    top_gainers = nse.nse_get_top_gainers().to_dict(orient="records")
    gainers = []
    for i in top_gainers:
        gainers.append({
            "name": i["meta"]["companyName"],
            "symbol": i["symbol"],
            "price": i["lastPrice"],
            "onedaychange": round(abs(float(i["change"])), 2),
            "onedaychangepercent": round(abs(float(i["pChange"])), 2),
            "positive": float(i["pChange"]) > 0,
            "logo": None        
        })

    return gainers

@app.get("/getTopLosers")
async def getTopLosers():
    top_losers = nse.nse_get_top_losers().to_dict(orient="records")
    losers = []
    for i in top_losers:
        losers.append({
            "name": i["meta"]["companyName"],
            "symbol": i["symbol"],
            "price": i["lastPrice"],
            "onedaychange": round(abs(float(i["change"])), 2),
            "onedaychangepercent": round(abs(float(i["pChange"])), 2),
            "positive": float(i["pChange"]) > 0,
            "logo": None        
        })

    return losers

@app.get("/getStock")
async def getStock(symbol: str):
    stock = nse.nse_eq(symbol)
    print(stock)
    logo = None
    with open("company_websites.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == symbol:
                website = row[1]
                break
    
    return {
        "name": stock["info"]["companyName"],
        "symbol": symbol,
        "price": stock["priceInfo"]["lastPrice"],
        "onedaychange": round(abs(float(stock["priceInfo"]["change"])), 2),
        "onedaychangepercent": round(abs(float(stock["priceInfo"]["pChange"])), 2),
        "positive": float(stock["priceInfo"]["pChange"]) > 0,
        "website": website,
        "industry": stock["industryInfo"]["industry"],
        }
    
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
    for holding in holdings:
        stock=nse.nse_eq(holding[0])
        totalbuy = holding[2] * holding[1]
        holdingsData["total"]["totalInvested"] += totalbuy
        holdingsData["total"]["totalCurrent"] += float(stock["priceInfo"]["lastPrice"]) * holding[1]
        holdingsData["holdings"].append({
        "name": stock["info"]["companyName"],
        "symbol": holding[0],
        "price": stock["priceInfo"]["lastPrice"],
        "quantity": holding[1],
        "buyPrice": holding[2],
        "currentPrice": stock["priceInfo"]["lastPrice"],
        "totalBuyPrice": totalbuy,
        "totalCurrentPrice": float(stock["priceInfo"]["lastPrice"]) * holding[1],
        "change": round(abs(totalbuy - float(stock["priceInfo"]["lastPrice"]) * holding[1]), 2),
        "changePercent": abs(round((float(stock["priceInfo"]["lastPrice"]) * holding[1] - totalbuy) / totalbuy * 100, 2)),
        "positive": float(stock["priceInfo"]["lastPrice"]) * holding[1] - totalbuy > 0,
        })
    
    if holdingsData["total"]["totalInvested"] != 0:
        holdingsData["total"]["totalChange"] = abs(holdingsData["total"]["totalCurrent"] - holdingsData["total"]["totalInvested"])
        holdingsData["total"]["totalChangePercent"] = abs(round((holdingsData["total"]["totalCurrent"] - holdingsData["total"]["totalInvested"]) / holdingsData["total"]["totalInvested"] * 100, 2))
        holdingsData["total"]["positive"] = holdingsData["total"]["totalCurrent"] - holdingsData["total"]["totalInvested"] > 0
    return holdingsData
    
    


## login stuff

@app.get("/checkUsername")
async def checkUsername(query: str):
    cur.execute(f"SELECT username FROM users WHERE username = \'{query.strip()}\'")
    if cur.rowcount > 0:
        return JSONResponse(status_code=400, content={"detail": "Username already taken"})
    else:
        return JSONResponse(status_code=200, content={"detail": "Username available"})

@app.post("/register")
async def register(data: dict):    
    
    password = data["password"]
    username = data["uname"]
    email = data["email"]
    cur.execute(f"SELECT username FROM users WHERE username = \'{username}\'")
    if cur.rowcount > 0:
        return JSONResponse(status_code=400, content={"detail": "Username already taken"})
    hashed_password = get_password_hash(password)
    print(hashed_password)
    cur.execute(f"INSERT INTO users (username, email, password) VALUES (\'{username}\', \'{email}\', \'{hashed_password}\')")
    uid = cur.lastrowid
    cur.execute(f"INSERT INTO wallet (userid) VALUES (\'{uid}\')")
    conn.commit()
    return JSONResponse(status_code=201, content={"detail": "Account registered successfully"})


@app.post("/login")
async def login(data: OAuth2PasswordRequestForm = Depends()):
    cur.execute(f"SELECT userid, password FROM users WHERE username = \'{data.username}\'")
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
    cur.execute(f"SELECT userid FROM users WHERE username = \'{username}\'")
    user = cur.fetchall()[0][0]
    cur.execute(f"SELECT balance FROM wallet WHERE userid = \'{user}\'")
    balance = cur.fetchall()[0]
    return {"balance": balance}


@app.post("/deposit")
async def deposit(data: dict):
    username = data["username"]
    amount = data["amount"]

    if amount <= 0:
        return JSONResponse(status_code=400, content={"detail": "Amount must be greater than zero"})
    
    cur.execute(f"SELECT userid FROM users WHERE username = \'{username}\'")
    user = cur.fetchone()
    if not user:
        return JSONResponse(status_code=404, content={"detail": "User not found"})
    
    cur.execute(f"UPDATE wallet SET balance = balance + {amount} WHERE userid = \'{user[0]}\'")
    conn.commit()
    return {"detail": "Deposit successful"}

@app.post("/withdraw")
async def withdraw(data: dict):
    username = data["username"]
    amount = data["amount"]
    if amount <= 0:
        return JSONResponse(status_code=400, content={"detail": "Amount must be greater than zero"})
    
    cur.execute(f"SELECT userid FROM users WHERE username = \'{username}\'")
    user = cur.fetchone()
    if not user:
        return JSONResponse(status_code=404, content={"detail": "User not found"})
    
    cur.execute(f"SELECT balance FROM wallet WHERE userid = \'{user[0]}\'")
    balance = cur.fetchone()[0]
    if balance < amount:
        return JSONResponse(status_code=400, content={"detail": "Insufficient funds"})
    
    cur.execute(f"UPDATE wallet SET balance = balance - {amount} WHERE userid = \'{user[0]}\'")
    conn.commit()
    return {"detail": "Withdrawal successful"}

