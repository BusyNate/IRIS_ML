"""
Bluetooth Receiver Script for Sleepiness Prediction Models

📌 Purpose:
- Connect to ESP32-C3 via Bluetooth (RFCOMM)
- Receive sensor data in format: "PD:<value>,MPU:<value>"
- Split into two data streams: photodiode and MPU6050
- Feed each stream into separate queues for ML model inference

🛠️ Requirements:
- ESP32-C3 must be paired with your PC
- ESP32 must transmit data as a string like: "PD:0.87,MPU:0.12"
- Install pybluez: pip install pybluez
"""

import bluetooth
import threading
import queue

# Queues to hold incoming sensor data for ML models
photodiode_queue = queue.Queue()
mpu6050_queue = queue.Queue()

# Replace this with your ESP32-C3's MAC address (check Bluetooth settings)
ESP32_MAC = "XX:XX:XX:XX:XX:XX"

def parse_data(data_str):
    """
    Parses incoming Bluetooth string into two float values.
    Expected format: "PD:<value>,MPU:<value>"
    Returns: (photodiode_value, mpu6050_value)
    """
    try:
        parts = data_str.strip().split(',')
        pd_val = float(parts[0].split(':')[1])
        mpu_val = float(parts[1].split(':')[1])
        return pd_val, mpu_val
    except Exception as e:
        print(f"[Parse Error] {e}")
        return None, None

def receiver_thread(sock):
    """
    Continuously reads data from Bluetooth socket and pushes values into queues.
    This runs in a background thread to avoid blocking your main ML pipeline.
    """
    while True:
        try:
            data = sock.recv(1024).decode('utf-8')
            pd_val, mpu_val = parse_data(data)
            if pd_val is not None and mpu_val is not None:
                photodiode_queue.put(pd_val)
                mpu6050_queue.put(mpu_val)
        except Exception as e:
            print(f"[Receive Error] {e}")
            break

def start_receiver():
    """
    Establishes Bluetooth connection and starts the receiver thread.
    Call this function once at the start of your ML pipeline or dashboard.
    """
    sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    sock.connect((ESP32_MAC, 1))
    print("[Connected] Bluetooth socket established.")
    thread = threading.Thread(target=receiver_thread, args=(sock,))
    thread.daemon = True
    thread.start()

# Optional: Run standalone for testing
if __name__ == "__main__":
    import time
    start_receiver()
    while True:
        if not photodiode_queue.empty():
            print("Photodiode:", photodiode_queue.get())
        if not mpu6050_queue.empty():
            print("MPU6050:", mpu6050_queue.get())
        time.sleep(0.1)
