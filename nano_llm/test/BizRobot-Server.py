import asyncio
import websockets

async def handle_client(websocket):
    print("Client connected")
    try:
        async for message in websocket:
            print(f"Received message: {message}")
    except websockets.ConnectionClosed:
        print("Client disconnected")

async def main():
    async with websockets.serve(handle_client, "localhost", 52560):
        print("Server running on ws://localhost:52560")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())

