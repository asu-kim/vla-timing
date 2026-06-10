import cv2
import pickle
import socket
import struct
import numpy as np

def receiver_start():

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #server_socket.bind(('localhost', 9999))
    #server_socket.listen(5)
    #print("Receiver is listening on port 9999...")

    conn, addr = server_socket.accept()
    print(f"Connection from {addr} has been established.")

    data_buffer = b""
    payload_size = struct.calcsize("Q")

    try:
        while True:
            while len(data_buffer) < payload_size:
                data_buffer += conn.recv(4096)

            packed_msg_size = data_buffer[:payload_size]
            data_buffer = data_buffer[payload_size:]
            msg_size = struct.unpack("Q", packed_msg_size)[0]

            while len(data_buffer) < msg_size:
                data_buffer += conn.recv(4096)

            frame_data = data_buffer[:msg_size]
            data_buffer = data_buffer[msg_size:]

            frame = pickle.loads(frame_data)
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

            #cv2.imshow('Received Video', frame)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break

            #Call the vla_inference code
            #vla_inference(frame)
            # Check for corner cases
            # Maybe I can use threads to queue the frames and process them in parallel

    except KeyboardInterrupt:
        print("Receiver is shutting down.")
    finally:
        conn.close()
        server_socket.close()
        cv2.destroyAllWindows()
        
#Will be removed later, just for testing purposes
def main():
    print()
    receiver_start()

#Will be removed later, just for testing purposes
if __name__ == "__main__":
    main()