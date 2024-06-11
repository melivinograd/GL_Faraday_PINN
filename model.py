import torch

class MLP(torch.nn.Module):
    """
    Multilayer Perceptron (MLP).

    This class defines a feedforward neural network with multiple linear hidden layers,
    hyperbolic tangent activation functions in the hidden layers, and a linear output layer.

    Args:
        sizes (list): List of integers specifying the number of neurons in each layer.
                      The first element should match the input dimension and the last
                      should match the output dimension.

    Attributes:
        layers (torch.nn.ModuleList): List containing the linear layers of the MLP.
        params (torch.nn.ParameterList): List containing additional parameters for the MLP.

    Methods:
        forward(x): Performs a forward pass through the MLP.

    Example:
        sizes = [input_dim, hidden1_dim, hidden2_dim, output_dim]
        mlp = MLP(sizes)
        input_tensor = torch.tensor([...])
        output = mlp(input_tensor)
    """
    def __init__(self, sizes: list[int]):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.params = torch.nn.ParameterList([
            torch.nn.Parameter(torch.rand(1), requires_grad=True) for _ in range(6)
        ])
        self.a = torch.nn.Parameter(torch.rand(1), requires_grad=True)  # for improved convergence

        for i in range(len(sizes) - 1):
            self.layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers[:-1]:
            h = torch.tanh(layer(h))
        output = self.layers[-1](h)
        return output

