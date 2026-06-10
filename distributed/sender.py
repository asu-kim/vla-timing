import cv2
import pickle
import socket
import struct
import time


def sender_start():

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('localhost', 9999))

    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            result, frame = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            data = pickle.dumps(frame)
            message_size = struct.pack("Q", len(data)) + data
            client_socket.sendall(message_size)

    except KeyboardInterrupt:
        print("Sender is shutting down.")
    finally:    
        cap.release()
        client_socket.close()

def main():
    sender_start()

if __name__ == "__main__":
    main()