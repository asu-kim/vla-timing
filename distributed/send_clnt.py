import cv2
import pickle
import socket
import struct
import time
import argparse
import os
from PIL import Image
import re

folder_path = 'images'

def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def connect_with_retry(ip, port, delay=2.0):
    """Keep trying until the sender is reachable. Handles 'Connection refused'."""
    while True:
        try:
            conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            conn.connect((ip, port))
            print(f"Connected to {ip}:{port}.")
            return conn
        except OSError as e:                       # ConnectionRefusedError is a subclass
            print(f"Connect failed ({e.__class__.__name__}); retrying in {delay:.0f}s...")
            time.sleep(delay)

def receiver_start(ip, port):

    while True:
        conn = connect_with_retry(ip, port)
        data_buffer = b""                          # fresh buffer every new connection
    
        try:
            files = sorted(os.listdir(folder_path), key=natural_key)
            for filename in files:
                if filename.endswith('.jpg'):
                    print(f"Sending {filename}...")
                    image = cv2.imread(os.path.join(folder_path, filename))
                    if image is None:                       # cv2.imread returns None on failure
                        print(f"Could not read {filename}, skipping.")
                        continue
                    ok, frame = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    data = pickle.dumps(frame)
                    message = struct.pack("Q", len(data)) + data   # 8-byte size header + payload
                    conn.sendall(message)
                else:
                    print(f"Skipping {filename}.")
        except KeyboardInterrupt:
            print("Sender is shutting down.")
        finally:
            conn.close()

def main():
    parser = argparse.ArgumentParser(description="Connect to a server and send data over a TCP socket.")
    parser.add_argument("ip", help="Server IP address (e.g. 127.0.0.1)")
    parser.add_argument("-p", "--port", type=int, default=8080, help="Server port (default: 8080)")
    args = parser.parse_args()
    
    receiver_start(args.ip, args.port)

if __name__ == "__main__":
    main()