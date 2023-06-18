# Log File:
# /Users/[your name]/Library/Preferences/Ableton/Live 11.2.6
import random
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from pythonosc import udp_client
import time

ip = "127.0.0.1"
to_ableton = 11000
TRACK = 0
CLIP = 0

p = {'name': ['kick', 'snare', 'hi-hat'],
     'points': [8, 16, 12],
     'start': [0, 3, 5],
     'size': 24,
     'length': 4}

def draw_clip(size, start, *instruments):
    client = udp_client.SimpleUDPClient(ip, to_ableton)
    client.send_message("/live/clip_slot/delete_clip", [TRACK, CLIP])
    client.send_message('/live/clip_slot/create_clip', [TRACK, CLIP, size / 4])
    #'/live/clip/remove/notes'
    for _ in range(4):
        time.sleep(0.1)
        for i in instruments[0]:
            client.send_message('/live/clip/add/notes', [TRACK, CLIP, 60, i / p['length'], 0.25, 100, 0])
        for i in instruments[1]:
            client.send_message('/live/clip/add/notes', [TRACK, CLIP, 63, i / p['length'], 1.25, 100, 0])
        for i in instruments[2]:
            client.send_message('/live/clip/add/notes', [TRACK, CLIP, 67, i / p['length'], 0.25, 110, 0])
    client.send_message('/live/clip/fire', [TRACK, CLIP])


def polygon(names, points, starts, size):
    d = {}
    for name, num_points, start, in zip(names, points, starts):
        if num_points == size:
            d[name] = list(range(size))
        else:
            d[name] = sorted([start] + random.sample(range(start+1, size), num_points-1))
    return d


def visualise(name, points, start, size):
    ax = plt.subplot(111, polar=True)
    total_points = np.linspace(0, 2 * pi, size, endpoint=False)
    ax.set_xticks(total_points)
    ax.set_xticklabels(range(size))
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[start[0]], labels[start[1]], labels[start[2]] = 'kick', 'snare', 'hi-hat'
    ax.set_xticklabels(labels)
    ax.tick_params(axis='x', which='major', pad=15)
    ax.set_theta_direction(-1)
    ax.set_theta_offset(pi / 2)
    plt.yticks([])
    x = []
    for i in range(len(points)):
        x.append(total_points[points[name[i]]])
    print(x)
    y = [1, ] * len(x[0])
    for i in range(len(name)):
        try:
            x[0] = np.append(x[0], x[0][0])
            y[0] = np.full(len(x[0]), 1)
            x[1] = np.append(x[1], x[1][0])
            y[1] = np.full(len(x[1]), 1)
            x[2] = np.append(x[2], x[2][0])
            y[2] = np.full(len(x[2]), 1)
        except IndexError:
            print("There's an Index Error")
            break
    for i in range(len(name)):
        ax.plot(x[i], y[i], linewidth=1, linestyle='solid', label=name[i])
    plt.ylim(0, 1)
    plt.legend(loc=(0.99, 0.95))
    plt.show()

    # Initialise the Instrument List
    instr_1, instr_2, instr_3 = None, None, None
    for i in range(len(name)):
        try:
            instr_1 = (points[name[0]])
            instr_2 = (points[name[1]])
            instr_3 = (points[name[2]])
        except KeyError:
            break
    draw_clip(size, start, instr_1, instr_2, instr_3)

d = polygon(p['name'], p['points'], p['start'], p['size'])
visualise(p['name'], d, p['start'], p['size'])
