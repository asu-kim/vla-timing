import cv2
import argparse
import numpy as np
from time import sleep
import socket, struct, pickle, time

from vla_inference_2 import load_items, start_vla_inference

payload_size = struct.calcsize("Q")

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

def recv_exactly(conn, n, buf):
    """Read until buf has >= n bytes. Returns (buf, ok); ok=False if the sender closed."""
    while len(buf) < n:
        chunk = conn.recv(4096)
        if not chunk:
            return buf, False
        buf += chunk
    return buf, True

def receiver_start(ip, port):

    processor, model = load_items()                # loaded ONCE, outside the loop

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((ip, port))
    server_socket.listen(5)
    print(f"Receiver listening on {ip}:{port}. Waiting for sender to connect...")
    conn, addr = server_socket.accept()
    print(f"Receiver connected from {addr}. Starting to receive data...")

    while True:
        #conn = connect_with_retry(ip, port)
        data_buffer = b""                          # fresh buffer every new connection
        try:
            while True:
                data_buffer, ok = recv_exactly(conn, payload_size, data_buffer)
                if not ok:
                    raise ConnectionError("sender closed")
                msg_size = struct.unpack("Q", data_buffer[:payload_size])[0]
                data_buffer = data_buffer[payload_size:]

                data_buffer, ok = recv_exactly(conn, msg_size, data_buffer)
                if not ok:
                    raise ConnectionError("sender closed mid-frame")
                frame_data = data_buffer[:msg_size]
                data_buffer = data_buffer[msg_size:]

                frame = pickle.loads(frame_data)

                #print(f"Frame hash: {do_hash(frame)}")

                start_vla_inference(processor, model, frame, "open the drawer")
        except ConnectionError as e:
            print(f"Disconnected: {e}. Reconnecting...")
        except KeyboardInterrupt:
            print("\nStopping.")
            conn.close()
            server_socket.close()
            return
        finally:
            conn.close()
            server_socket.close()

def main():
    parser = argparse.ArgumentParser(description="Connect to a server and send data over a TCP socket.")
    parser.add_argument("ip", help="Server IP address (e.g. 127.0.0.1)")
    parser.add_argument("-p", "--port", type=int, default=8080, help="Server port (default: 8080)")
    args = parser.parse_args()

    receiver_start(args.ip, args.port)

if __name__ == "__main__":
    main()

#################################################
# Test - Will be removed.
#################################################

'''
import hashlib

def do_hash(data):
    if hasattr(data, "tobytes"):
        return hashlib.md5(data.tobytes()).hexdigest()
    return hashlib.md5(data).hexdigest()
'''