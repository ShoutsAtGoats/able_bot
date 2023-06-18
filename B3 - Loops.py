import rtmidi
import time

midiout = rtmidi.MidiOut()
ports = midiout.get_ports()
print(ports)
midiout.open_port(5)

tempo = 0.4

# with midiout:
#     for bar in range(4):
#         for note in range(4):
#             note_on = [0x90, 60, 50]
#             note_off = [0x80, 60, 0]
#             midiout.send_message(note_on)
#             time.sleep(tempo)
#             midiout.send_message(note_off)
#         for note in range(4):
#             note_on = [0x90, 62, 50]
#             note_off = [0x80, 62, 0]
#             midiout.send_message(note_on)
#             time.sleep(tempo)
#             midiout.send_message(note_off)
#         for note in range(4):
#             note_on = [0x90, 67, 50]
#             note_off = [0x80, 67, 0]
#             midiout.send_message(note_on)
#             time.sleep(tempo)
#             midiout.send_message(note_off)
#         for note in range(4):
#             note_on = [0x90, 58, 50]
#             note_off = [0x80, 58, 0]
#             midiout.send_message(note_on)
#             time.sleep(tempo)
#             midiout.send_message(note_off)

names = ['Alex', 'Jane', 'Johnny', 'Tristan', 'Judy']
for name in names:
    print(name)

