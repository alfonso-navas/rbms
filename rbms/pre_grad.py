import torch
from torch import Tensor
from torch.optim import Optimizer
from rbms.classes import EBM

class L1Regularization(torch.nn.Module):
    def __init__(self, optimizer: list[Optimizer], lambda_l1: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = optimizer
        self.lambda_l1 = lambda_l1
        self.penalty: dict[int, Tensor] = {}

    def forward(self, input):
        self.penalty = {}
        for opt in self.optimizer:
            for p in opt.param_groups[0]["params"]:
                curr_penalty = -self.lambda_l1 * torch.sign(p)
                p.grad += curr_penalty
                self.penalty[id(p)] = curr_penalty.detach()

class L2Regularization(torch.nn.Module):
    def __init__(self, optimizer: list[Optimizer], lambda_l2: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = optimizer
        self.lambda_l2 = lambda_l2
        self.penalty: dict[int, Tensor] = {}

    def forward(self, input):
        self.penalty = {}
        for opt in self.optimizer:
            for p in opt.param_groups[0]["params"]:
                curr_penalty = -self.lambda_l2 * p
                p.grad += curr_penalty
                self.penalty[id(p)] = curr_penalty.detach()

# Only implemented for Ising-Ising
class EffectiveL2Regularization(torch.nn.Module):
    def __init__(self, optimizer: list[Optimizer], lambda_eff_l2: float, *args, **kwargs):
        # super().__init__(*args, **kwargs)
        super().__init__()
        self.optimizer = optimizer
        self.lambda_eff_l2 = lambda_eff_l2
        self.model: EBM = kwargs["model"]
        self.batch_size = kwargs["batch_size"]
        self.penalty: dict[int, Tensor] = {}
    
    def forward(self, input):
        self.penalty = {}
        v = 2*torch.randint(0, 2, (self.batch_size,self.model.num_visibles), device=self.model.device, dtype=self.model.dtype) - 1
        energy = self.model.compute_energy_visibles(v)
        energy_gradient = self.model.compute_gradient_energy_visibles(v)
        param_index = {id(p): i for i, p in enumerate(self.model.parameters())}

        for opt in self.optimizer:
            for p in opt.param_groups[0]["params"]:
                i = param_index[id(p)]
                aux_energy = energy.clone()
                for _ in range(energy_gradient[i].dim() - 1):
                    aux_energy = aux_energy.unsqueeze(-1)
                penalty = ((aux_energy - aux_energy.mean(axis=0, keepdim=True))
                           *(energy_gradient[i] - energy_gradient[i].mean(axis=0, keepdim=True))
                )

                curr_penalty = -self.lambda_eff_l2 * penalty.mean(axis=0)
                p.grad += curr_penalty
                self.penalty[id(p)] = curr_penalty.detach()



class ClipGradNorm(torch.nn.Module):
    def __init__(self, optimizer: list[Optimizer], max_grad_norm, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm

    def forward(self, input):
        for opt in self.optimizer:
            torch.nn.utils.clip_grad_norm_(
                opt.param_groups[0]["params"], max_norm=self.max_grad_norm
            )


class NormalizeGrad(torch.nn.Module):
    def __init__(self, optimizer: list[Optimizer], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = optimizer

    def forward(self, input):
        for opt in self.optimizer:
            norm_grad = torch.nn.utils.get_total_norm(
                [p.grad for p in opt.param_groups[0]["params"] if p.grad is not None]
            )
            for p in opt.param_groups[0]["params"]:
                p.grad /= norm_grad


def build_pre_grad_update(
    optimizer: list[Optimizer],
    lambda_l1: float,
    lambda_l2: float,
    normalize_grad: bool,
    max_grad_norm: float,
    lambda_eff_l2: float | None = 0.0,
    **kwargs,
) -> torch.nn.Sequential:
    """Build the sequence of modules applied to the gradient before the optimizer step.

    Args:
        optimizer (list[Optimizer]): The optimizers holding the parameters.
        lambda_l1 (float): Strength of the L1 regularization.
        lambda_l2 (float): Strength of the L2 regularization.
        normalize_grad (bool): Whether to normalize the gradient.
        max_grad_norm (float): Clip the gradient norm. Non-positive disables clipping.
        lambda_eff_l2 (float, optional): Strength of the effective L2 regularization.
            Only implemented for Ising-Ising models. Defaults to 0.0.

    Keyword Args:
        model (EBM): The model. Only required when `lambda_eff_l2 > 0`.
        batch_size (int): Number of random configurations drawn to estimate the
            effective L2 penalty. Only required when `lambda_eff_l2 > 0`.

    Returns:
        torch.nn.Sequential: The modules to apply, in order.

    Notes:
        - Modules are instantiated lazily: `model` and `batch_size` are only needed
          when the effective L2 regularization is actually active.
    """
    lambda_l1 = lambda_l1 or 0.0
    lambda_l2 = lambda_l2 or 0.0
    lambda_eff_l2 = lambda_eff_l2 or 0.0
    max_grad_norm = -1 if max_grad_norm is None else max_grad_norm

    modules: list[torch.nn.Module] = []
    if lambda_l1 > 0:
        modules.append(L1Regularization(optimizer=optimizer, lambda_l1=lambda_l1))
    if lambda_l2 > 0:
        modules.append(L2Regularization(optimizer=optimizer, lambda_l2=lambda_l2))
    if lambda_eff_l2 > 0:
        missing = [k for k in ("model", "batch_size") if k not in kwargs]
        if missing:
            raise ValueError(
                f"The effective L2 regularization requires {missing} to be passed to "
                "`build_pre_grad_update`."
            )
        modules.append(
            EffectiveL2Regularization(
                optimizer=optimizer,
                lambda_eff_l2=lambda_eff_l2,
                model=kwargs["model"],
                batch_size=kwargs["batch_size"],
            )
        )
    if normalize_grad:
        modules.append(NormalizeGrad(optimizer=optimizer))
    if max_grad_norm > 0:
        modules.append(ClipGradNorm(optimizer=optimizer, max_grad_norm=max_grad_norm))
    return torch.nn.Sequential(*modules)

def get_penalty(
    pre_grad_update: torch.nn.Sequential, params: EBM
) -> dict[str, Tensor]:
    """Total penalty added to the gradient by the modules of `pre_grad_update`
    during the last call, for each parameter of the model.

    Args:
        pre_grad_update (torch.nn.Sequential): The modules applied to the gradient
            before the optimizer step.
        params (EBM): The model whose parameters received the penalty.

    Returns:
        dict[str, Tensor]: A mapping name -> penalty tensor, with the same names and
            shapes as `params.named_parameters()`.

    Notes:
        - Sign convention: the returned tensors are the terms *added* to the gradient.
          With no rescaling module active, `p.grad = grad_log_likelihood + penalty`.
        - `NormalizeGrad` and `ClipGradNorm` are not penalties: they rescale the whole
          gradient and are therefore not accounted for here.
    """
    named_tensors = params.named_parameters_tensor()
    id_to_name = {id(p): name for name, p in named_tensors.items()}
    total = {name: torch.zeros_like(p) for name, p in named_tensors.items()}
    for module in pre_grad_update:
        penalty = getattr(module, "penalty", None)
        if not penalty:
            continue
        for param_id, value in penalty.items():
            name = id_to_name.get(param_id)
            if name is None:
                raise RuntimeError(
                    "A penalty was recorded for a tensor which is not a parameter of "
                    "the model passed to `get_penalty`."
                )
            total[name] += value
    return total