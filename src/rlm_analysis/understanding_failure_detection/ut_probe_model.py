import torch.nn as nn

hs_dict = {
        "Qwen/Qwen3-4B": 2560,
        "openai/gpt-oss-20b": 2880
    }

class MLPProbe(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=2):
        super(MLPProbe, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)  # No Sigmoid here
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)  # Raw logits
        return x

class LinearProbe(nn.Module):
    def __init__(self, input_size, output_size=2):
        super(LinearProbe, self).__init__()
        self.output = nn.Linear(input_size, output_size)  # No Sigmoid here

    def forward(self, x):
        x = self.output(x)  # Raw logits
        return x

def load_model(input_size, hidden_size, output_size, ckpt_weights=None):
    if hidden_size==0:
        model = LinearProbe(input_size, output_size)
    else:
        model = MLPProbe(input_size, hidden_size, output_size)
    if ckpt_weights is not None:
        model.load_state_dict(ckpt_weights)
    return model
