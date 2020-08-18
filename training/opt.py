import torch
# Optimizer
# We used the Adam optimizer (cite) with $\beta_1 = 0.9, \beta2 = 0.98, \epsilon=10^{-9}$. We varied the
# learning rate over the course of training, according to the formula $lrate = d_{transformer}^{-0.5} \cdot \min(step_num^{-0.5}, step_num \cdot warmup_steps^{-1.5}$
# This corresponds to increasing the learning rate linearly for the first warmup_steps training steps, and decreasing it thereafter
# proportionally to the inverse square root of the step number. We used warmup_steps=4000$.

# Note: This part is very important. Need to train with this setup of the transformer.

class NoamOpt:
    "Optim wrapper that implements rate"
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


    # TODO UNTESTED
    def load(self, model_path: str):
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        state_dict = params['state_dict']
        self.optimizer.load_state_dict(state_dict)
        opt = NoamOpt(model_size=args['model_size'], factor=args['factor'], warmup=args['warmup'],
                   optimizer=adam)

        transformer_model.load_state_dict(params['state_dict'])

    # TODO UNTESTED
    def save(self, path: str):
        print(f"Save optimizer params to {path}.")
        params = {
            'args': dict(_step = self._step, warmup = self.warmup, factor = self.factor, model_size = self.model_size, _rate = self._rate),
            'state_dict': self.optimizer.state_dict()
        }
        torch.save(params, path)

def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr = 0, betas=(0.9, 0.98), eps=1e-9))