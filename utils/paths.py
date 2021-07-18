import os
from pathlib import Path

class NetPaths:
    """Manages and configures the paths used by WaveRNN, Tacotron, and the data."""
    def __init__(self, net_id):
        self.base = Path(__file__).parent.parent.expanduser().resolve()

        # Network Paths
        self.checkpoints = self.base/'checkpoints'/f'{net_id}.dnn'
        self.latest_weights = self.checkpoints/'latest_weights.pyt'
        self.latest_optim = self.checkpoints/'latest_optim.pyt'
        self.step = self.checkpoints/'step.npy'
        self.log = self.checkpoints/'log.txt'

        self.create_paths()

    def create_paths(self):
        os.makedirs(self.checkpoints, exist_ok=True)

    def get_net_named_weights(self, name):
        """Gets the path for the weights in a named net checkpoint."""
        return self.checkpoints/f'{name}_weights.pyt'

    def get_net_named_optim(self, name):
        """Gets the path for the optimizer state in a named net checkpoint."""
        return self.checkpoints/f'{name}_optim.pyt'

