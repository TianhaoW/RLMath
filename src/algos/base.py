class RLAlgo:
    def __init__(self, config, env, model, device, logger):
        self.config = config
        self.env = env
        self.model = model
        self.device = device
        self.logger = logger

    def train(self):
        raise NotImplementedError("Subclasses must implement `train()` method.")

    def test(self):
        raise NotImplementedError("Subclasses must implement `test()` method.")