import asyncio
import csv
from bleak import BleakClient

# Replace with your ESP32's actual MAC address
ESP32_ADDRESS = "xx:xx:xx:xx:xx:xx" 
UART_TX_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"  # Notify characteristic

def handle_rx(_, data: bytearray):
    try:
        decoded = data.decode().strip()
        parts = decoded.split(",")
        if len(parts) == 8:
            timestamp, photo, ax, ay, az, gx, gy, gz = map(float, parts)
            print(f"[{int(timestamp)} ms] Photo: {photo} | Accel: ({ax}, {ay}, {az}) | Gyro: ({gx}, {gy}, {gz})")

            ## Store data

            with open("./data/raw/sensor_raw.csv","a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, photo, ax, ay, az, gx, gy, gz])
        else:
            print("⚠️ Unexpected payload format:", decoded)
    except Exception as e:
        print("❌ Error parsing data:", e)

async def main():
    async with BleakClient(ESP32_ADDRESS) as client:
        print("🔗 Connected to ESP32")
        await client.start_notify(UART_TX_UUID, handle_rx)
        print("📡 Listening for sensor data...")
        while True:
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
