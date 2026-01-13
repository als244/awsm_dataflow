import time
import torch
import uuid

class Sequence:
    def __init__(self, tokens, targets=None, weights=None, loss_function=None, seq_id=None):
        self.seq_id = seq_id if seq_id is not None else str(uuid.uuid4())
        self.tokens = tokens
        self.targets = targets
        self.weights = weights
        #if self.weights is None:
        #    self.weights = torch.ones(len(self.tokens), dtype=torch.float32)
        self.loss_function = loss_function
        self.per_token_loss = torch.zeros(len(self.tokens), dtype=torch.float32, device="cpu", pin_memory=True)
        self.create_time = time.time()
        self.start_train_time = None
        self.complete_train_time = None

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        return self.tokens[index]

    def __setitem__(self, index, value):
        self.tokens[index] = value
