import torch
from torch.utils.data import TensorDataset, DataLoader
from model import MLP
import pickle

def load_training_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def generate_data_for_net(rout, iout, interval, N):
    # Sample interval, transpose so that the time steps become the rows
    testr = torch.tensor(rout[:, ::interval]).T
    testi = torch.tensor(iout[:, ::interval]).T

    #Concatenate real and imag, add dim so it's 3D (time, space, channels)
    data_y = torch.cat((testr[:, :, None], testi[:, :, None]), dim=-1)

    # Time and space grids
    t_data = torch.linspace(-1, 1, data_y.shape[0]).view(-1)
    x_data = torch.linspace(-1, 1, N)
    x_grid, t_grid = torch.meshgrid(x_data, t_data, indexing='ij')
    x_grid = x_grid.T[:, :, None].requires_grad_(True)
    t_grid = t_grid.T[:, :, None].requires_grad_(True)
    input_data = torch.cat((x_grid, t_grid), dim=-1)

    return input_data.to(torch.float32), data_y.to(torch.float32), x_grid, t_grid

def create_dataloader(input_data, data_y, batch_size=100):
    dataset = TensorDataset(input_data, data_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def train_model(model, dataloader, optimizer, lossfun, epochs=1):
    for epoch in range(epochs):
        for input, target in dataloader:
            optimizer.zero_grad()
            outputs = model(input)
            loss = lossfun(target, outputs)
            loss.backward()
            optimizer.step()

        with torch.autograd.no_grad():
            print(f"Epoch {epoch + 1}/{epochs}\tTraining Loss: {loss.item():.6f}")

def train_model_with_physics(model, input_data, data_y, x_grid, t_grid, optimizer, epochs=1, lambda_weight=0.5):
    for epoch in range(epochs):
        optimizer.zero_grad()

        yh = model(input_data)
        loss1 = torch.mean((yh - data_y)**2)

        dxr  = torch.autograd.grad(yh[:, :, 0], x_grid, torch.ones_like(yh[:, :, 0]), create_graph=True)[0]
        dxxr = torch.autograd.grad(dxr, x_grid, torch.ones_like(dxr), create_graph=True)[0]
        dtr  = torch.autograd.grad(yh[:, :, 0], t_grid, torch.ones_like(yh[:, :, 0]), create_graph=True)[0]
        dxr  = dxr[:, :, 0]
        dxxr = dxxr[:, :, 0]
        dtr  = dtr[:, :, 0]

        dxi  = torch.autograd.grad(yh[:, :, 1], x_grid, torch.ones_like(yh[:, :, 1]), create_graph=True)[0]
        dxxi = torch.autograd.grad(dxi, x_grid, torch.ones_like(dxi), create_graph=True)[0]
        dti  = torch.autograd.grad(yh[:, :, 1], t_grid, torch.ones_like(yh[:, :, 1]), create_graph=True)[0]
        dxi  = dxi[:, :, 0]
        dxxi = dxxi[:, :, 0]
        dti  = dti[:, :, 0]

        sqr = yh[:, :, 0]**2 + yh[:, :, 1]**2

        physicsr = dtr - model.p1 * yh[:, :, 0] + model.p2 * dxr + model.p3 * (dxxr - model.p4 * dxxi) - model.p5 * sqr * (yh[:, :, 0] - model.p6 * yh[:, :, 1])
        physicsi = dti - model.p1 * yh[:, :, 1] + model.p2 * dxi + model.p3 * (dxxi + model.p4 * dxxr) - model.p5 * sqr * (yh[:, :, 1] + model.p6 * yh[:, :, 0])
        physics = physicsr + physicsi
        loss2 = lambda_weight * torch.mean((physics)**2)

        loss = loss1 + loss2
        loss.backward()
        optimizer.step()

        with torch.autograd.no_grad():
            print(f"Epoch {epoch + 1}/{epochs}\tData Loss: {loss1.item():.6f}\tPDE Loss: {loss2.item():.6f}\tTraining Loss: {loss.item():.6f}")


if __name__ == "__main__":
    # Parameters
    interval = 100
    input_dim = 2
    hidden_layers = [128] * 6
    output_dim = 2
    batch_size = 1
    learning_rate = 1e-4
    initial_epochs = 1  # Initial training with MSE loss
    physics_epochs = 5  # Further training with physics-informed loss
    lambda_weight = 0.5

    # Load training data
    training_data_file = './datasets/training_data.pkl'
    training_data = load_training_data(training_data_file)

    # Assuming you want to use the first dataset entry for training
    params, rout, iout = training_data[0]
    N = rout.shape[0]

    # Generate the data
    input_data, data_y, x_grid, t_grid = generate_data_for_net(rout, iout, interval, N)

    # Create DataLoader
    dataloader = create_dataloader(input_data, data_y, batch_size)

    # Define the network
    model = MLP([input_dim] + hidden_layers + [output_dim])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lossfun = torch.nn.MSELoss()

    # Initial training with MSE loss
    print("Starting initial training with MSE loss...")
    train_model(model, dataloader, optimizer, lossfun, epochs=initial_epochs)

    #  Further training with physics-informed loss
    print("Starting further training with physics-informed loss...")
    train_model_with_physics(model, input_data, data_y, x_grid, t_grid, optimizer, epochs=physics_epochs, lambda_weight=lambda_weight)
