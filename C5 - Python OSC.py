# Log File:
# /Users/jor/Library/Preferences/Ableton/Live 11.2.6
from pythonosc import udp_client
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

import threading
import time

ip = "127.0.0.1"
to_ableton = 11000
from_ableton = 11001

def serve():
    def print_handler(address, *args):
        print(args)
    dispatcher = Dispatcher()
    dispatcher.map("*", print_handler)
    server = BlockingOSCUDPServer((ip, from_ableton), dispatcher)
    time.sleep(0.1)
    server.serve_forever()  #Blocks

def client():
    client = udp_client.SimpleUDPClient(ip, to_ableton)
    client.send_message("Hello Ableton, how are you?", None)
    #client.send_message('/live/song/get/tempo', None)
    time.sleep(0.1)

t1 = threading.Thread(target=serve)
t1.start()
client()

