# Docs
# https://github.com/ideoforms/AbletonOSC
# Log File:
# /Users/jor/Library/Preferences/Ableton/Live 11.2.6
# https://github.com/ideoforms/AbletonOSC
from pythonosc import udp_client
import time
import sys

IP = '127.0.0.1'
TO_ABLETON = 11000
TEMPO = 90
TRACK = 4

crotchet = TEMPO / 60
quaver = crotchet / 2
semiquaver = quaver / 2


msg = {'metro': '/live/song/set/metronome',
       'set_tempo': '/live/song/set/tempo',
       'play_clip': '/live/clip/fire'}

client = udp_client.SimpleUDPClient(IP, TO_ABLETON)

order = [1, 4, 8, 2, 5, 5, 6, 3, 7, 2, 9]

def send_msg():
    client.send_message(msg['set_tempo'], TEMPO)
    for clip_number in order:
        client.send_message(msg['play_clip'], [TRACK, clip_number])
        time.sleep(crotchet)

send_msg()
sys.exit()
