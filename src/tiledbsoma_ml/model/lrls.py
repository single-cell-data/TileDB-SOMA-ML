from torch import sigmoid
from torch.nn import Linear, Module, ReLU


class LRLS(Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        middle_dim: int | None = None,
    ):
        super().__init__()
        middle_dim = middle_dim or input_dim
        self.net1 = Linear(input_dim, middle_dim)
        self.relu = ReLU()
        self.net2 = Linear(middle_dim, output_dim)

    def forward(self, x):
        return sigmoid(self.net2(self.relu(self.net1(x))))
