import pickle
import matplotlib.pyplot as plt
import numpy as np

def load_dataset(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def plot_spatiotemporal(data, title):
    for params, rout, iout in data[:2]:  # Displaying only first 2 entries for brevity
        field_intensity = rout**2 + iout**2
        plt.figure(figsize=(10, 6))
        plt.imshow(field_intensity.T, aspect='auto', extent=[0, field_intensity.shape[1], 0, field_intensity.shape[0]])
        plt.colorbar()
        plt.xlabel('Time Step')
        plt.ylabel('Spatial Coordinate')
        plt.title(f'{title}')
        print(title, params)
        plt.show()

if __name__ == "__main__":
    training_data_file = './datasets/training_data.pkl'
    testing_data_file = './datasets/testing_data.pkl'

    training_data = load_dataset(training_data_file)
    testing_data = load_dataset(testing_data_file)

    print("Training Data:")
    plot_spatiotemporal(training_data, 'Training Data')

    print("Testing Data:")
    plot_spatiotemporal(testing_data, 'Testing Data')

