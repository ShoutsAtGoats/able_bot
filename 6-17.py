import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import sklearn.neural_network
import sklearn.neural_network._multilayer_perceptron as mlp
# from sklearn.neural_network import MLPClassifier
from pythonosc import osc_server, udp_client
import os
from midi import generate_midi
from osc import send_midi
import numpy as np

class AbletonAgent:
    def __init__(self, sub_agent_limit, iterations):
        self.sub_agent_limit = sub_agent_limit
        self.iterations = iterations

    def generate_song(self, prompt):
        # Parse the prompt to determine the required song elements
        song_elements = self.parse_prompt(prompt)

        # Generate a list of sub-agents based on the song elements
        sub_agents = self.generate_sub_agents(song_elements)

        # Initialize the best cost to an arbitrarily large number
        # best_cost = float('inf')
        best_song = None
        best_cost = 1.0
        threshold = 0.33

        # Iterate for a certain number of iterations
        for i in range(self.iterations):
            # Generate a song by combining the output of each sub-agent
            song = self.generate_combined_song(sub_agents)

            # Calculate the cost of the song
            cost = self.calculate_cost(song)
            assert isinstance(best_cost, float)
            assert isinstance(cost, float)
            # If this song is better than the current best, update the best
            if cost < best_cost:
                best_cost = cost
                best_song = song

            # If the cost is low enough, we're satisfied and can return the song
            if cost < threshold:
                return best_song

        # If no satisfactory song was found after all iterations, return the best one we found
        return best_song

    

    def generate_sub_agents(self, song_elements):
        sub_agents = []
        for element in song_elements:
            sub_agent = OSCNeuralNet()
            sub_agent.train(*self.get_training_data_for_element(element))
            sub_agents.append(sub_agent)
        return sub_agents

    def parse_prompt(self, prompt):
        # TODO: Implement this method to parse the prompt and return a list of required song elements
        pass

    def get_training_data_for_element(self, element):
        if element == 'drums':
            X_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
            y_train = [0, 1, 1, 0]
        elif element == 'lead':
            X_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
            y_train = [0, 0, 0, 1]
        else:
            raise ValueError(f"Invalid song element: {element}")
        return X_train, y_train

    def generate_combined_song(self, sub_agents):
        # TODO: Implement this method to generate a song by combining the output of each sub-agent
        pass

    def calculate_cost(self, song):
        # TODO: Implement this method to calculate the cost of a song
        pass


class OSCNeuralNet:
    def __init__(self, in_address="localhost", in_port=5005, out_address="localhost", out_port=5006):
        self.client = udp_client.SimpleUDPClient(out_address, out_port)
        self.server = osc_server.ThreadingOSCUDPServer((in_address, in_port), self.osc_dispatcher())

        self.clf = mlp(hidden_layer_sizes=(100,), max_iter=500, alpha=1e-4,
                                 solver='sgd', verbose=10, tol=1e-4, random_state=1,
                                 learning_rate_init=.1)

    def osc_dispatcher(self):
        dispatcher = osc_server.Dispatcher()
        dispatcher.map("/nn/input", self.handle_osc_message)
        return dispatcher

    def osc_to_input_vector(self, osc_message):
        return np.array([osc_message.args])

    def output_vector_to_osc(self, output_vector):
        return udp_client.OscMessageBuilder(address="/nn/output").add_arg(output_vector).build()

    def handle_osc_message(self, addr, *args):
        input_vector = self.osc_to_input_vector(args)
        prediction = self.clf.predict(input_vector)
        osc_message = self.output_vector_to_osc(prediction)
        self.client.send(osc_message)

    def train(self, X_train, y_train):
        self.clf.fit(X_train, y_train)

    def start(self):
        self.server.serve_forever()

class AbletonModel:
    def __init__(self):
        self.drums = OSCNeuralNet(in_port=5005, out_port=5007)
        self.lead = OSCNeuralNet(in_port=5006, out_port=5008)

    def train(self, X_train_drums, y_train_drums, X_train_lead, y_train_lead):
        self.drums.train(X_train_drums, y_train_drums)
        self.lead.train(X_train_lead, y_train_lead)

    def start(self):
        self.drums.start()
        self.lead.start()

class AbletonAgent:
    def __init__(self, sub_agent_limit, iterations):
        self.sub_agent_limit = sub_agent_limit
        self.iterations = iterations

    def generate_song(self, prompt):
        # Parse the prompt to determine the required song elements
        song_elements = self.parse_prompt(prompt)

        # Generate a list of sub-agents based on the song elements
        sub_agents = self.generate_sub_agents(song_elements)

        # Initialize the best cost to an arbitrarily large number
        # best_cost = float('inf')
        best_song = None
        best_cost = 1.0
        threshold = 0.33

        # Iterate for a certain number of iterations
        for i in range(self.iterations):
            # Generate a song by combining the output of each sub-agent
            song = self.generate_combined_song(sub_agents)

            # Calculate the cost of the song
            cost = self.calculate_cost(song)
            assert isinstance(best_cost, float)
            assert isinstance(cost, float)
            # If this song is better than the current best, update the best
            if cost < best_cost:
                best_cost = cost
                best_song = song

            # If the cost is low enough, we're satisfied and can return the song
            if cost < threshold:
                return best_song

        # If no satisfactory song was found after all iterations, return the best one we found
        return best_song

    def generate_combined_song(self, sub_agents):
        # Generate a MIDI file for each sub-agent
        midi_files = []
        for sub_agent in sub_agents:
            midi_file = generate_midi(sub_agent)
            midi_files.append(midi_file)

        # Send the MIDI files to Ableton Live using OSC messages
        for i, midi_file in enumerate(midi_files):
            track_number = i + 1
            send_midi(midi_file, track_number)

        # TODO: Implement this method to generate a song by combining the output of each sub-agent
        pass

    # ...