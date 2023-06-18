# Goal: Reset device racks with a specific name
# 1. Get the total number of tracks, so that we can loop over it
# 2. Create a dictionary per track to store the device names, and the device index
# 3. Create a list to the 'per track' dictionaries (so we get a list of dicts)
# 4. Loop over the full list and compare the names to see if there is a match
# 5. When there is a match, reset the devices based on the track and the device index

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient
import sys


ip = "127.0.0.1"
to_ableton = 11000
from_ableton = 11001


class AbletonAPI:
    def __init__(self):
        self._server_return_value = None
        def store_results(address_name, *args):
            self._server_return_value = args
        dispatcher = Dispatcher()
        dispatcher.map("*", store_results)
        self.client = SimpleUDPClient(ip, to_ableton)
        self.server = BlockingOSCUDPServer((ip, from_ableton), dispatcher)

    def call(self, func_name, *args):
        self.client.send_message(func_name, args)
        self.server.handle_request()
        return self._server_return_value


a = AbletonAPI()
num_tracks = a.call('/live/song/get/num_tracks')[0]
all_track_devs = []

for track in range(num_tracks):
    dev_dic = {}
    devices = a.call("/live/track/get/devices/name", track)[1:]
    for dev_idx, dev in enumerate(devices):
        if dev not in dev_dic:
            dev_dic[dev] = []
        dev_dic[dev].append(dev_idx)
    all_track_devs.append(dev_dic)


def does_device_exist(name):
    print(f"Looking for a device called {name}")
    client = SimpleUDPClient(ip, to_ableton)
    for track, devices in enumerate(all_track_devs):
        if name in devices:
            dev_idx = devices[name][0]
            dev_type = a.call("/live/device/get/class_name", track, dev_idx)[2]
            #print(dev_type)
            #print(dev_idx)
            if dev_type == 'AudioEffectGroupDevice':
                current_values = a.call("/live/device/get/parameters/value", track, dev_idx)
                print("Currently the device has these values", current_values)
                client.send_message("/live/device/set/parameters/value", [track, devices[name][0],  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    print("*" * 20 + f" Devices Named {name} have been cleared " + "*" * 20)


does_device_exist('RESET')
sys.exit()

