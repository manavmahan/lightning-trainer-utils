import torch


class EMA:
    def __init__(self, parameters, update_after_step=100, inv_gamma=1.0, power=0.75, min_value=0.0, max_value=0.9999):
        self.shadow = {}
        self.backup = {}
        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value
        self.step_count = 0

        for name, param in parameters:
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def _get_decay(self):
        step = max(0, self.step_count - self.update_after_step)
        if step <= 0:
            return 0.0
        decay = 1.0 - (1.0 + step / self.inv_gamma) ** -self.power
        return max(self.min_value, min(decay, self.max_value))

    @torch.no_grad()
    def update(self, named_params):
        decay = self._get_decay()
        self.decay = decay
        for name, param in named_params:
            if name not in self.shadow:
                self.shadow[name] = param.data.clone().detach().to(param.device)
            else:
                self.shadow[name] = (
                    self.shadow[name].to(param.device)
                    .mul(decay)
                    .add(param.data, alpha=1.0 - decay)
                )
        self.step_count += 1

    def apply_to(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {
            "shadow": self.shadow,
            "step_count": self.step_count,
            "update_after_step": self.update_after_step,
            "inv_gamma": self.inv_gamma,
            "power": self.power,
            "min_value": self.min_value,
            "max_value": self.max_value
        }

    def load_state_dict(self, state_dict):
        self.shadow = state_dict["shadow"]
        self.step_count = state_dict["step_count"]
        self.update_after_step = state_dict["update_after_step"]
        self.inv_gamma = state_dict["inv_gamma"]
        self.power = state_dict["power"]
        self.min_value = state_dict["min_value"]
        self.max_value = state_dict["max_value"]