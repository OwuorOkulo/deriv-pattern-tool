import asyncio
import json
import websockets
import pandas as pd
from detector import run_all_detectors

DERIV_WS_URL = "wss://ws.binaryws.com/websockets/v3?app_id=1089"

async def get_candles(symbol, granularity, count=500):
    async with websockets.connect(DERIV_WS_URL) as ws:
        payload = {
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": count,
            "end": "latest",
            "granularity": granularity,
            "style": "candles"
        }
        await ws.send(json.dumps(payload))
        response = json.loads(await ws.recv())
        if "error" in response:
            print("Error:", response["error"]["message"])
            return []
        return response["candles"]

def build_dataframe(candles):
    df = pd.DataFrame(candles)
    df["time"] = pd.to_datetime(df["epoch"], unit="s")
    df = df[["time", "open", "high", "low", "close"]]
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    df = df.reset_index(drop=True)
    return df

async def main():
    print("Connecting to Deriv API...")
    candles = await get_candles(symbol="R_75", granularity=60, count=500)
    df = build_dataframe(candles)
    print(f"Fetched {len(df)} candles")
    run_all_detectors(df)

asyncio.run(main())