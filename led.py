WLED_IP = "10.0.0.99"
WLED_PORT = 21324
NUM_LEDS = 200
import socket

def send_wled_states(on):
    rgb = []
    for c in on:
        rgb.append([255] * 3 if c else [0] * 3)

    send_wled_rgb(rgb)

def send_wled_rgb(rgb):
    packet = bytearray([2, 5])
    for c in rgb:
        packet.extend(c)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(packet, (WLED_IP, WLED_PORT))
    except socket.error as e:
        print(f"Socket error: {e}")
    finally:
        sock.close()