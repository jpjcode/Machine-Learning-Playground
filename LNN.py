import torch
from torch import nn
import torch.nn.functional as F
import SpikingNeurons as SN
"""
This is suppose to be a type of machine learning algorithm called Liquid Neural Networks. It is not
 well optimize and needs some refactory.
 
 Note: Liquid State Machines operate the same as a RNN except the only weight that is under backpropagation is the output weight.
 It also has spiking Neurons
"""


class LiquidStateMachine(nn.Module):
    def __init__(self, input_size, output_size, layer):

        """
        :param input_size: size of the inputs in a 1 by input_size shape
        :param output_size: size of the outputs in a 1 by output_size shape
        :param layer: array of integer that represent the amount of neurons per layer
        """
        super(LiquidStateMachine, self).__init__()

        #input weights for each layer
        self.input_weights = []

        #hidden states
        self.states = []

        #reservoir weights and biases
        self.reservoir_weights = []
        self.reservoir_biases = []

        #output weight (keep in mind that this is the only parameter that is bring trained)
        self.output_weight = nn.Parameter(torch.randn(output_size, output_size), requires_grad=True)

        #Spiking neurons
        self.SNCs = []

        #Type of spiking neurons (not yet fully  implemented)
        self.snType = SN.LIFCell

        self.init_nerons(input_size, output_size, layer)

    #This function to initialize neurons, might not be good practice but I put it there so it easier to read.
    def init_nerons(self, input_size, output_size, layer):

        #topology is the layer with the input size at the begining
        topology = []
        topology.append(input_size)
        topology.extend(layer)

        for i in range(1, len(topology)):
            #adds states
            self.states.append(torch.randn(1, topology[i]))

            #adds spiking neurons
            self.SNCs.append(SN.LIFCell(topology[i], 0.2, 0.01))

            #adds biases
            self.reservoir_biases.append(torch.zeros(1, topology[i]))

            #takes to account of matrix multiplication
            prev = topology[i - 1]
            current = topology[i]

            #adds input and reservoir weights in account to matrix mulitplication
            self.input_weights.append(torch.randn(prev, current))
            self.reservoir_weights.append(torch.randn(current, current))

        #output weights but has nn.Parameter so that it is the only weight being trained
        self.output_weight = nn.Parameter(torch.randn(topology[-1], output_size), requires_grad=True)

    def forward(self, input):

        #forward pass
        weighted_input = torch.matmul(input, self.input_weights[0])

        self.states[0] = F.tanh(
            weighted_input + torch.matmul(self.SNCs[0].forward(weighted_input), self.reservoir_weights[0]) +
            self.reservoir_biases[0])

        for i in range(1, len(self.states)):
            weighted_input = torch.matmul(self.states[i - 1], self.input_weights[i])
            self.states[i] = F.tanh(
                weighted_input + torch.matmul(self.SNCs[i].forward(weighted_input), self.reservoir_weights[i]) +
                self.reservoir_biases[i])

        return torch.matmul(self.states[-1], self.output_weight)
