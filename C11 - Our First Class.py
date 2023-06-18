# Classes
# Object Oriented Programming (OOP)
# self, instances, methods, constructor
# Instance Variables Vs. Class Variables
# /Users/jor/Library/Preferences/Ableton/Live 11.2.6
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient
import sys


# class Person:
#     def __init__(self, name, age, arms):
#         self.name = name
#         self.age = age
#         self.arms = arms
#
#     def print_stuff(self, stuff):
#         print(stuff)
#
# a = Person('Jor', 32, 2)
# b = Person('Mike', 62, 3)
# print(a)
# print(b)
# print(a.name)
# print(b.age)
#
# a.print_stuff("hello how are you?", 9000)

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
        self.server.handle_request()                                                        # Blocks less
        return self._server_return_value

a = AbletonAPI()

num_tracks = a.call('/live/song/get/num_tracks')
print(num_tracks[0])

