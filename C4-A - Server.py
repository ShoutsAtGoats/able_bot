# Run this in Terminal to see the open ports
# lsof -Pn -i4

import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((socket.gethostname(), 7777))
s.listen(5)

while True:
    clientsocket, adr = s.accept()
    clientsocket.send("You are now connected to the server".encode())


