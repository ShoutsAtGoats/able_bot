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
CLIP = 8

crotchet = TEMPO / 60
quaver = crotchet / 2
semiquaver = quaver / 2


msg = {'metro': '/live/song/set/metronome',
       'set_tempo': '/live/song/set/tempo',
       'play_clip': '/live/clip/fire',
       'pitch_coarse': '/live/clip/set/pitch_coarse'
       }

pitches = [0, 2, 0, 5, 0, 7, 0, 3, 0, 12] * 128
client = udp_client.SimpleUDPClient(IP, TO_ABLETON)

def send_msg():
    client.send_message(msg['set_tempo'], TEMPO)
    for pitch in pitches:
        client.send_message(msg['play_clip'], [TRACK, CLIP])
        client.send_message(msg['pitch_coarse'], [TRACK, CLIP, pitch * 6])
        time.sleep(semiquaver / 4)

send_msg()
sys.exit()
