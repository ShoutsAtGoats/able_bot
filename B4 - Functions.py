import rtmidi
import time

midiout = rtmidi.MidiOut()
midiout.open_port(5)

def send_notes(pitch=60, repeat=4):
    for note in range(repeat):
        note_on = [0x90, pitch, 80]
        note_off = [0x80, pitch, 0]
        midiout.send_message(note_on)
        time.sleep(0.5)
        midiout.send_message(note_off)

with midiout:
    for i in range(4):
        send_notes(60, 2)
        send_notes(63, 2)
        send_notes(60, 2)
        send_notes(65, 2)
        send_notes(60, 2)
        send_notes(67, 2)