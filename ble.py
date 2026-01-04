import asyncio
import os
from bleak import BleakClient

ADDRESS = "18:93:D7:0A:A3:67"
CHARACTERISTIC_UUID = "0000ffe5-0000-1000-8000-00805f9b34fb" 
FILE_PATH = r"data.txt"

print(os.path.exists(FILE_PATH))

def notification_handler(sender, data):
    decoded = data.decode("utf-8", errors="ignore").strip()
    if decoded:
        print(f"Received: {decoded}")
        with open(FILE_PATH, "a", encoding="utf-8") as f:
            f.write(decoded + "\n")
            f.flush() 

async def main(address):
    print(f"Attempting to connect to {address}...")
    
    async with BleakClient(address, timeout=20.0) as client:
        if not client.is_connected:
            print("Failed to connect")
            return

        print(f"Connected! Listening for data on {CHARACTERISTIC_UUID}...")

        await client.start_notify(CHARACTERISTIC_UUID, notification_handler)

        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            if client.is_connected:
                await client.stop_notify(CHARACTERISTIC_UUID)
                print("Disconnected safely.")

if __name__ == "__main__":
    asyncio.run(main(ADDRESS))