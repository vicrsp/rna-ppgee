import torch
import torch.nn.functional as F
import warnings

from torch.serialization import SourceChangeWarning

warnings.filterwarnings("ignore", category=SourceChangeWarning)

class MLP(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layer1 = torch.nn.Linear(self.input_dim, self.hidden_dim)
        torch.manual_seed(2)
        self.layer1.weight.data.uniform_(-1.0, 1.0)
        self.tanh = torch.nn.Tanh()
        self.layer2 = torch.nn.Linear(self.hidden_dim, self.output_dim)
        torch.manual_seed(2)
        self.layer2.weight.data.uniform_(-1.0, 1.0)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

    def forward(self, x, calc_gram=False):
        x = self.layer1(x)
        x = self.tanh(x)
        x = self.layer2(x)
        output = self.sigmoid(x)
        output = F.normalize(output, dim=1)

        return output


model = torch.load('model.pth')
data = torch.tensor([[1., 0.]], dtype=torch.float32)

output = model(data)

print(output)
