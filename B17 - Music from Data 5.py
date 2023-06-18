# Dataset Documentation
# https://www.kaggle.com/datasets/markmarkoh/near-earth-asteroids

# Velocity Infinity          -> Pitch               -> Higher Speed         -> High Pitch
# Minimum Distance           -> Note Length         -> Further Distance     -> Longer Note
# ref                        -> Velocity            -> Higher Ref           -> Higher Velocity

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from midiutil import MIDIFile


def convert_data(row):
    converted = np.array([])
    for i in row:
        stripped = i.split('/')[0]
        converted = np.append(converted, float(stripped))
    return converted


def read_and_plot(file, r1, r2, r3):
    d = pd.read_csv(file)
    pitch = d[r1].values
    length = d[r2].values
    velocity = d[r3].values
    if isinstance(pitch[0], str):
        pitch = convert_data(pitch)
    if isinstance(length[0], str):
        length = convert_data(length)
    if isinstance(velocity[0], str):
        velocity = convert_data(velocity)
    return pitch, length, velocity

pitch, length, velocity = read_and_plot('./00data/near_earth.csv', 'Vinfinity(km/s)', 'CA DistanceMinimum(LD/AU)', 'Vrelative(km/s)')
print(pitch)

# ================= CONVERT RANGE FUNCTION =================
def convert_range(value, in_min, in_max, out_min, out_max):
    return out_min + (value - in_min) / (in_max - in_min) * (out_max - out_min)

# ================= CONVERT LENGTH (x) =================
kms_per_beat = 8.0
scaled_x = [i * kms_per_beat for i in length]
x = sorted(scaled_x, reverse=False)

# ================= CONVERT PITCH (y) =================
y = convert_range(pitch, min(pitch), max(pitch), 0, 1)

# ================= CONVERT VELOCITY (v) =================
v = convert_range(velocity, min(velocity), max(velocity), 0, 127)
v = [round(v) for v in v]

def note_to_number(note):
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = 0
    if len(note) == 2:
        octave = int(note[1])
        note = NOTES.index(note[0])
    elif len(note) == 3:
        octave = int(note[2])
        note = NOTES.index(note[0:2])
    note += (12 * octave)
    return note


our_scale = ['C2', 'D2', 'D#2', 'F2', 'G2', 'G#2', 'B2',
            'C3', 'D3', 'D#3', 'F3', 'G3', 'G#3', 'B3',
            'C4', 'D4', 'D#4', 'F4', 'G4', 'G#4', 'B4',
            'C5', 'D5', 'D#5', 'F5', 'G5', 'G#5', 'B5',]

note_midi = [note_to_number(n) for n in our_scale]

len_notes = len(note_midi)
midi_nn = []
for index in range(len(x)):
    note_index = round(convert_range(y[index], 0, 1, len_notes-1, 0))
    midi_nn.append(note_midi[note_index])

plt.scatter(x, midi_nn, s=50 * y, c=v)
plt.xlabel('Time (Beats)')
plt.ylabel('MIDI Note Number')
plt.show()

BPM = 80

midi_file = MIDIFile(1)
midi_file.addTempo(track=0, time=0, tempo=BPM)

for i in range(len(x)):
    midi_file.addNote(track=0, channel=0, pitch=midi_nn[i], time=x[i], duration=0.5, volume=v[i])

with open('astroids-1.mid', 'wb') as f:
    midi_file.writeFile(f)



