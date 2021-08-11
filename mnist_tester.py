from typing import Type, Dict, Tuple, List
from time import perf_counter
from mnist_reader import get_data, render_mnist_image
import sys
from collections import defaultdict
import numpy as np

sys.path.append("ann")
from network import Network
from layer import Layer

def main():
    print("loading data")
    train_images, train_labels, test_images, test_labels = get_data()
    print("data loaded")
    print("train_images.shape", train_images.shape)
    print("train_labels.shape", train_labels.shape)
    i = np.random.randint(0, len(train_images))
    print("random training image", i, ":")
    print(render_mnist_image(train_images[i]))
    print("random training image value:", train_labels[i])

    input_feature_count = len(train_images[0])
    print("pixels per image:", input_feature_count)
    output_feature_count = len(train_labels[0])
    print("possible values:", output_feature_count)

    neuron_counts_in_hidden_layers = [800, 128, 48]
    epoch_count = 10000
    learning_rate = 0.0625

    def test(activation: Type[Layer.Activation]) -> Tuple[float, float]:
        """
        returns accuracy, training time
        """
        net = Network(input_feature_count)
        for neuron_count in neuron_counts_in_hidden_layers:
            net.add_layer(neuron_count, activation)
        net.add_layer(output_feature_count, Layer.Sigmoid)

        t0 = perf_counter()
        train_end_error = net.train(train_images, train_labels, epoch_count, learning_rate, 4000)
        t1 = perf_counter()
        train_time = t1 - t0
        print("train end error value:", train_end_error)

        output = net.predict(test_images)
        values: np.ndarray = np.argmax(output, axis = 1)  # type: ignore
        assert len(values) == len(test_labels)
        # print(output[:3])
        # print(values[:3])

        """
        # random result from tests
        i = np.random.randint(0, len(test_images))
        print("random test image", i, ":")
        print(render_mnist_image(test_images[i]))
        print("random test image label:", test_labels[i])
        print("neural network predicted value:", values[i], "from", output[i])
        """

        correct_count: int = 0
        for i in range(len(values)):
            correct_count += values[i] == test_labels[i]
        
        return correct_count / len(output), train_time
    
    sample_count = 20

    accuracies: Dict[Type[Layer.Activation], List[float]] = defaultdict(list)
    times: Dict[Type[Layer.Activation], List[float]] = defaultdict(list)

    activations = (
        Layer.SQRT,
        Layer.TruncatedSQRT,
        Layer.Swish,
        Layer.ReLU,
        Layer.Sigmoid,
        Layer.TanH
    )

    for i in range(sample_count):
        print("running sample", i)
        for activation in activations:
            print("activation", activation)
            accuracy, training_time = test(activation)
            times[activation].append(training_time)
            accuracies[activation].append(accuracy)
    print(sample_count, "samples")
    for activation in activations:
        print("  result", activation)
        print("    accuracy rate:", sum(accuracies[activation]) / sample_count)
        print("    accuracies:")
        print(accuracies[activation])
        print("    average time:", sum(times[activation]) / sample_count)
        print("    times:")
        print(times[activation])

if __name__ == "__main__":
    main()
