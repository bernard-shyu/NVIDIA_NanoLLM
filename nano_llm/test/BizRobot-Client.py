import asyncio
import websockets

async def send_commands(command):
    uri = "ws://localhost:52560"

    async with websockets.connect(uri) as websocket:
        await websocket.send(command)
        print(f"Sent: {command}")

if __name__ == "__main__":
    commands = ["Command 1", "Command 2", "Command 3"]
    for command in commands:
        #asyncio.run(send_commands())
        asyncio.run(send_commands(command))

