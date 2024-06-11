import matplotlib.pyplot as plt
import os
import pickle
import torch

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import MLP

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

def train_model(model, dataloader, optimizer, lossfun, epochs=1, writer=None):
    for epoch in range(epochs):
        for batch_idx, (input, target) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(input)
            loss = lossfun(outputs, target)
            loss.backward()
            optimizer.step()

            if writer:
                writer.add_scalar('Training Loss', loss.item(), epoch)

                if epoch % 10 == 0:
                    plot_reconstructions_to_tensorboard(writer, input, target,
                                                        model, epoch,
                                                        prefix='Initial_')

        with torch.autograd.no_grad():
            print(f"Epoch {epoch + 1}/{epochs}\tTraining Loss: {loss.item():.6f}")

def train_model_with_physics(model, input_data, data_y, x_grid, t_grid, optimizer, epochs=1, lambda_weight=0.5, writer=None):
    for epoch in range(epochs):
        optimizer.zero_grad()

        yh = model(input_data)
        loss1 = torch.mean((yh - data_y) ** 2)

        dxr = torch.autograd.grad(outputs=yh[:, :, 0], inputs=x_grid, grad_outputs=torch.ones_like(yh[:, :, 0]), create_graph=True)[0]
        dxxr = torch.autograd.grad(outputs=dxr, inputs=x_grid, grad_outputs=torch.ones_like(dxr), create_graph=True)[0]
        dtr = torch.autograd.grad(outputs=yh[:, :, 0], inputs=t_grid, grad_outputs=torch.ones_like(yh[:, :, 0]), create_graph=True)[0]
        dxr, dxxr, dtr = dxr[:, :, 0], dxxr[:, :, 0], dtr[:, :, 0]

        dxi = torch.autograd.grad(outputs=yh[:, :, 1], inputs=x_grid, grad_outputs=torch.ones_like(yh[:, :, 1]), create_graph=True)[0]
        dxxi = torch.autograd.grad(outputs=dxi, inputs=x_grid, grad_outputs=torch.ones_like(dxi), create_graph=True)[0]
        dti = torch.autograd.grad(outputs=yh[:, :, 1], inputs=t_grid, grad_outputs=torch.ones_like(yh[:, :, 1]), create_graph=True)[0]
        dxi, dxxi, dti = dxi[:, :, 0], dxxi[:, :, 0], dti[:, :, 0]

        sqr = yh[:, :, 0] ** 2 + yh[:, :, 1] ** 2

        physicsr = dtr - model.p1 * yh[:, :, 0] + model.p2 * dxr + model.p3 * (dxxr - model.p4 * dxxi) - model.p5 * sqr * (yh[:, :, 0] - model.p6 * yh[:, :, 1])
        physicsi = dti - model.p1 * yh[:, :, 1] + model.p2 * dxi + model.p3 * (dxxi + model.p4 * dxxr) - model.p5 * sqr * (yh[:, :, 1] + model.p6 * yh[:, :, 0])
        physics = physicsr + physicsi
        loss2 = lambda_weight * torch.mean((physics) ** 2)

        loss = loss1 + loss2
        loss.backward()
        optimizer.step()

        if writer:
            writer.add_scalar('Data Loss', loss1.item(), epoch)
            writer.add_scalar('PDE Loss', loss2.item(), epoch)
            writer.add_scalar('Total Loss', loss.item(), epoch)

            # Log parameters p1 to p6
            writer.add_scalar('Parameters/p1', model.p1.item(), epoch)
            writer.add_scalar('Parameters/p2', model.p2.item(), epoch)
            writer.add_scalar('Parameters/p3', model.p3.item(), epoch)
            writer.add_scalar('Parameters/p4', model.p4.item(), epoch)
            writer.add_scalar('Parameters/p5', model.p5.item(), epoch)
            writer.add_scalar('Parameters/p6', model.p6.item(), epoch)

            if epoch % 10 == 0:  # Save reconstructions every 10 epochs
                plot_reconstructions_to_tensorboard(writer, input_data,
                                                    data_y, model, epoch,
                                                    prefix='Physics_')

        with torch.autograd.no_grad():
            print(f"Epoch {epoch + 1}/{epochs}\tData Loss: {loss1.item():.6f}\tPDE Loss: {loss2.item():.6f}\tTotal Loss: {loss.item():.6f}")

        torch.cuda.empty_cache()

def plot_reconstructions_to_tensorboard(writer, input_data, data_y, model, epoch, prefix=''):
    with torch.no_grad():
        yh = model(input_data)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(data_y[:, :, 0].cpu(), aspect='auto', origin='lower')
        ax[0].set_title('True Real Part')
        ax[1].imshow(yh[:, :, 0].cpu(), aspect='auto', origin='lower')
        ax[1].set_title('Predicted Real Part')
        plt.suptitle(f'{prefix} Epoch {epoch}')
        writer.add_figure(f'{prefix}Reconstructions', fig, global_step=epoch)
        plt.close(fig)

def get_next_log_dir(base_dir='./logs'):
    existing_dirs = os.listdir(base_dir)
    existing_ints = [int(d.split('_')[-1]) for d in existing_dirs if d.split('_')[-1].isdigit()]
    next_int = max(existing_ints, default=0) + 1
    return f'{base_dir}/run_{next_int}'

if __name__ == "__main__":

    interval = 100
    input_dim = 2
    hidden_layers = [128] * 6
    output_dim = 2
    batch_size = 10
    learning_rate = 1e-4
    initial_epochs = 30  # Initial training with MSE loss
    physics_epochs = 20  # Further training with physics-informed loss
    lambda_weight = 0.5

    training_data_file = './datasets/training_data.pkl'
    training_data = load_training_data(training_data_file)

    params, rout, iout = training_data[0]
    N = rout.shape[0]

    input_data, data_y, x_grid, t_grid = generate_data_for_net(rout, iout, interval, N)
    dataloader = create_dataloader(input_data, data_y, batch_size)

    model = MLP([input_dim] + hidden_layers + [output_dim])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lossfun = torch.nn.MSELoss()

    # Create a unique log directory
    log_dir = get_next_log_dir()
    writer = SummaryWriter(log_dir=log_dir)

    print("Starting initial training with MSE loss...")
    train_model(model, dataloader, optimizer, lossfun, epochs=initial_epochs, writer=writer)

    print("Starting further training with physics-informed loss...")
    train_model_with_physics(model, input_data, data_y, x_grid, t_grid, optimizer, epochs=physics_epochs, lambda_weight=lambda_weight, writer=writer)

    writer.close()

    model_save_path = './models'
    os.makedirs(model_save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_save_path, 'model.pth'))

