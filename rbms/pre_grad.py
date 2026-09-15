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
                p.grad += self.penalty
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
    lambda_eff_l2 : float,
    normalize_grad: bool,
    max_grad_norm: float,
    **kwargs,
):
    # return torch.compile(
    #     torch.nn.Sequential(
    #         *[L1Regularization(optimizer=optimizer, lambda_l1=lambda_l1)]
    #         * (lambda_l1 > 0),
    #         *[L2Regularization(optimizer=optimizer, lambda_l2=lambda_l2)]
    #         * (lambda_l2 > 0),
    #         *[EffectiveL2Regularization(optimizer=optimizer, lambda_eff_l2=lambda_eff_l2, 
    #                                     model=kwargs["model"], batch_size = kwargs["batch_size"])]
    #         * (lambda_eff_l2 > 0),
    #         *[NormalizeGrad(optimizer=optimizer)] * normalize_grad,
    #         *[ClipGradNorm(optimizer=optimizer, max_grad_norm=max_grad_norm)]
    #         * (max_grad_norm > 0),
    #     )
    # )
    return torch.nn.Sequential(
        *[L1Regularization(optimizer=optimizer, lambda_l1=lambda_l1)]
        * (lambda_l1 > 0),
        *[L2Regularization(optimizer=optimizer, lambda_l2=lambda_l2)]
        * (lambda_l2 > 0),
        *[EffectiveL2Regularization(optimizer=optimizer, lambda_eff_l2=lambda_eff_l2, 
                                    model=kwargs["model"], batch_size = kwargs["batch_size"])]
        * (lambda_eff_l2 > 0),
        *[NormalizeGrad(optimizer=optimizer)] * normalize_grad,
        *[ClipGradNorm(optimizer=optimizer, max_grad_norm=max_grad_norm)]
        * (max_grad_norm > 0),
        )

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