from typing import Type, List
import numpy as np
from time import perf_counter
import sys

sys.path.append("ann")
from layer import Layer
from network import Network

def test(activation: Type[Layer.Activation]):
    # 4 input sets, 2 input features
    input_sets = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # an output for each input set to train on
    target_output = np.array([[0], [1], [1], [0]])  # xor
    input_feature_count = len(input_sets[0])
    output_feature_count = len(target_output[0])

    neuron_counts_in_hidden_layers = [5, 3]
    hidden_activation = activation
    epoch_count = 40000
    learning_rate = 0.0625

    net = Network(input_feature_count)
    for neuron_count in neuron_counts_in_hidden_layers:
        net.add_layer(neuron_count, hidden_activation)
    net.add_layer(output_feature_count, Layer.Sigmoid)

    end_error = net.train(input_sets, target_output, epoch_count, learning_rate, False)

    # print(net)
    # print("results:")
    # print(net.predict(test_input))
    return end_error

def main():
    test_count = 100

    for activation in (Layer.SQRT, Layer.TruncatedSQRT, Layer.Swish, Layer.ReLU, Layer.Sigmoid, Layer.TanH):
        print("testing", activation)
        total = 0
        times: List[float] = []
        for _ in range(test_count):
            t0 = perf_counter()
            error = test(activation)
            t1 = perf_counter()
            times.append(t1- t0)
            if error > 0.1:
                print("failure with error:", error)
                total += 1
        print("result", activation)
        print("fail rate:", total / test_count)
        print("time:", sum(times) / test_count)


if __name__ == "__main__":
    main()
