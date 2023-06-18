# Docs
# https://github.com/ideoforms/AbletonOSC
# Log File:
# /Users/jor/Library/Preferences/Ableton/Live 11.2.6
# https://github.com/ideoforms/AbletonOSC
from pythonosc import udp_client
import time
import sys

ip = '127.0.0.1'
to_ableton = 11000

msg = {'metro': '/live/song/set/metronome',
       'set_tempo': '/live/song/set/tempo'}

client = udp_client.SimpleUDPClient(ip, to_ableton)
client.send_message(msg['set_tempo'], 80)

def send_msg():
    for _ in range(8):
        client.send_message(msg['metro'], 1)
        time.sleep(0.5)
        client.send_message(msg['metro'], 0)
        time.sleep(0.5)

send_msg()
sys.exit()