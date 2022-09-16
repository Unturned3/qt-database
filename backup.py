
import torch
from torch import nn
import copy

def clone(obj):
    return copy.deepcopy(obj)

class FCN_200(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        # out = (in - kern) / stride + 1
        # input: 1x200
        # output neuron should have a receptive field size of 150 or more

        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=3, padding=0),   # 66
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=0),   # 64 
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=0),   # 62
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 31
            
            nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=0),   # 14
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=0),   # 12
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=0),   # 10
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 5

            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=0),   # 1
        )

        # conv.out: 64x1

        f_prob = nn.Sequential(
            nn.Conv1d(64, 2, kernel_size=1),
            nn.Softmax(dim=1),
        )
        f_peak = nn.Conv1d(64, 1, kernel_size=1)

        #self.p_prob = clone(f_prob)
        #self.p_peak = clone(f_peak)
        self.q_prob = clone(f_prob)
        self.q_peak = clone(f_peak)
        #self.t_prob = clone(f_prob)
        #self.t_peak = clone(f_peak)

        """
        self.pps = nn.Sequential(
            nn.Conv1d(64, 6, kernel_size=1),
            nn.LeakyReLU(),
        )
        """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        q_prob = self.q_prob(x)
        q_peak = self.q_peak(x)
        return q_prob.view(q_prob.size(0), 2), q_peak.view(q_peak.size(0), 1)

