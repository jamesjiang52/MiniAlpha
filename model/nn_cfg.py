import yaml


class NNConfig:
    """
    Convenience class so we don't have to index a dictionary
    """
    def __init__(self, cfg):
        self.cfg = cfg

    def __getattr__(self, name):
        return self.cfg[name]


def load_cfg(path):
    """
    Load a config from the specified path
    """
    with open(path, 'r') as f:
        return NNConfig(yaml.safe_load(f))
