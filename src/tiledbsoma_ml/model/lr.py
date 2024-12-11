from torch import sigmoid
from torch.nn import Linear, Module


class LogisticRegression(Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()  # noqa: UP008
        self.linear = Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = sigmoid(self.linear(x))
        return outputs
