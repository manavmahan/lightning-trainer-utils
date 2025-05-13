import copy
import torch


class EMA:
    def __init__(
        self,
        model,
        update_after_step=100,
        inv_gamma=1.0,
        power=3 / 4,
        min_value=0.0,
        max_value=0.9999,
    ):
        """
        Initialize the EMAWarmup instance.

        Args:
            model: The model whose parameters will be averaged.
            update_after_step (int): The step after which EMA updates begin.
            inv_gamma (float): Inverse gamma value for decay calculation.
            power (float): Power for the decay calculation.
            min_value (float): Minimum decay value.
            max_value (float): Maximum decay value.
        """
        self.model = copy.deepcopy(model).eval()
        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value
        self.optimization_step = 0
        self.shadow_params = {
            name: param.clone() for name, param in model.named_parameters()
        }
        self.collected_params = None

    def get_decay(self, step):
        """
        Calculate the EMA decay factor for a given step.

        Args:
            step (int): The current optimization step.

        Returns:
            float: The decay factor.
        """
        step = max(0, step - self.update_after_step)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power
        if step <= 0:
            return 0.0
        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model):
        """
        Update the EMA with the current parameters of the model.

        Args:
            new_model: The current model with parameters to be averaged.
        """
        self.decay = self.get_decay(self.optimization_step)
        for name, param in new_model.named_parameters():
            ema_param = self.shadow_params.get(name, torch.tensor(0)).to(param.device)
            self.shadow_params[name] = ema_param.mul(self.decay).add(
                param.data, alpha=(1 - self.decay)
            )
        self.optimization_step += 1

    def store(self, parameters):
        """
        Store the current parameters for restoring later.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be updated with the stored parameters.
        """
        if self.collected_params is None:
            raise RuntimeError("No `store()`ed weights to `restore()`")
        for stored_param, param in zip(self.collected_params, parameters):
            param.data.copy_(stored_param.data)

    def state_dict(self):
        """
        Returns the state of the EMA as a dict.

        Returns:
            dict: EMA state.
        """
        return {
            "min_decay": self.min_value,
            "optimization_step": self.optimization_step,
            "update_after_step": self.update_after_step,
            "use_ema_warmup": True,
            "inv_gamma": self.inv_gamma,
            "power": self.power,
            "shadow_params": self.shadow_params,
            "collected_params": self.collected_params,
        }

    def load_state_dict(self, state_dict):
        """
        Load the EMA state.

        Args:
            state_dict (dict): EMA state dictionary.
        """
        state_dict = copy.deepcopy(state_dict)
        self.min_value = state_dict.get("min_decay", self.min_value)
        self.optimization_step = state_dict.get(
            "optimization_step", self.optimization_step
        )
        self.update_after_step = state_dict.get(
            "update_after_step", self.update_after_step
        )
        self.inv_gamma = state_dict.get("inv_gamma", self.inv_gamma)
        self.power = state_dict.get("power", self.power)
        self.shadow_params = state_dict["shadow_params"]
        self.collected_params = state_dict.get("collected_params", None)


# Usage example:
# model = ...  # your PyTorch model
# ema = EMAWarmup(model)
# for epoch in range(num_epochs):
#     train(model)
#     ema.step(model)
