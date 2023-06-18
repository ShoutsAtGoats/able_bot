# Docs
# https://github.com/ideoforms/AbletonOSC
# Log File:
# /Users/jor/Library/Preferences/Ableton/Live 11.2.6
# https://github.com/ideoforms/AbletonOSC
# The Dispatcher uses an adaptation of Regex Patterns
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc import udp_client
import threading
import time
import sys

ip = '127.0.0.1'
to_ableton = 11000
from_ableton = 11001
TRACK = 1

track_name = []
num_scenes = []
scenes = []

def serve():
    def print_handler(address, *args):
        if address == '/live/track/get/name':
            track_name.append(args[0])
            track_name.append(args[1])
        elif address == '/live/song/get/num_scenes':
            num_scenes.append(args[0])
        elif address == '/live/clip_slot/get/has_clip':
            scenes.append(args[0])
    dispatcher = Dispatcher()
    dispatcher.map("*", print_handler)
    server = BlockingOSCUDPServer((ip, from_ableton), dispatcher)
    time.sleep(0.1)
    server.serve_forever()                  # Block forever

def client():
    client = udp_client.SimpleUDPClient(ip, to_ableton)
    # Find the name of the track specified
    client.send_message('/live/track/get/name', TRACK)
    time.sleep(0.1)
    # Get the total number of scenes in the Project
    client.send_message('/live/song/get/num_scenes', None)
    time.sleep(0.1)
    # See if the track we want to operate on has clips
    for i in range(num_scenes[0]):
        client.send_message('/live/clip_slot/get/has_clip', [TRACK, i])
    time.sleep(0.1)
    idx = 0
    for j, i in enumerate(scenes):
        if i == True:
            time.sleep(0.1)
            client.send_message('/live/clip/set/name', [TRACK, j, track_name[1] + '-' + str(idx) + 'A'])
            idx += 1

t1 = threading.Thread(target=serve)
t1.start()
client()
sys.exit()



