import asyncio
import json
import websockets

DERIV_WS_URL = "wss://ws.binaryws.com/websockets/v3?app_id=1089"

class DerivClient:
    def __init__(self, api_token=None):
        self.api_token = api_token
        self.ws = None

    async def connect(self):
        self.ws = await websockets.connect(DERIV_WS_URL)
        print("Connected to Deriv API")
        if self.api_token:
            await self.authorize()

    async def authorize(self):
        payload = {"authorize": self.api_token}
        await self.ws.send(json.dumps(payload))
        response = json.loads(await self.ws.recv())
        if "error" in response:
            raise Exception(f"Auth failed: {response['error']['message']}")
        print("Authorized successfully")

    async def get_candles(self, symbol, granularity, count=500):
        payload = {
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": count,
            "end": "latest",
            "granularity": granularity,
            "style": "candles"
        }
        await self.ws.send(json.dumps(payload))
        response = json.loads(await self.ws.recv())
        if "error" in response:
            raise Exception(f"Data fetch failed: {response['error']['message']}")
        return response["candles"]

    async def disconnect(self):
        if self.ws:
            await self.ws.close()
            print("Disconnected")