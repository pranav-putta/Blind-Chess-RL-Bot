import yaml
from util import Configuration


def load_config():
    global ENV
    with open('config.yaml') as f:
        ENV = Configuration(**yaml.safe_load(f))
