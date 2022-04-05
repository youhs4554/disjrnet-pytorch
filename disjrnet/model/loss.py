"""
Code from
    https://github.com/idocx/BP_MLL_Pytorch/blob/master/bp_mll.py
"""

import torch
from torch import Tensor


class BPMLLLoss(torch.nn.Module):
    def __init__(self, bias=(1, 1), eps=1e-7):
        super(BPMLLLoss, self).__init__()
        self.bias = bias
        self.eps = eps  # to prevent zero-division
        assert len(self.bias) == 2 and all(map(lambda x: isinstance(x, int) and x > 0, bias)), \
            "bias must be positive integers"

    def forward(self, c: Tensor, y: Tensor) -> Tensor:
        r"""
        compute the loss, which has the form:
        L = \sum_{i=1}^{m} \frac{1}{|Y_i| \cdot |\bar{Y}_i|} \sum_{(k, l) \in Y_i \times \bar{Y}_i} \exp{-c^i_k+c^i_l}
        :param c: prediction tensor, size: batch_size * n_labels
        :param y: target tensor, size: batch_size * n_labels
        :return: size: scalar tensor
        """
        y = y.float()
        y_bar = -y + 1
        y_norm = torch.pow(y.sum(dim=(1,)), self.bias[0])
        y_bar_norm = torch.pow(y_bar.sum(dim=(1,)), self.bias[1])
        return torch.mean(1 / (torch.mul(y_norm, y_bar_norm)+self.eps) * self.pairwise_sub_exp(y, y_bar, c))

    def pairwise_sub_exp(self, y: Tensor, y_bar: Tensor, c: Tensor) -> Tensor:
        r"""
        compute \sum_{(k, l) \in Y_i \times \bar{Y}_i} \exp{-c^i_k+c^i_l}
        """
        truth_matrix = y.unsqueeze(2).float() @ y_bar.unsqueeze(1).float()
        exp_matrix = torch.exp(c.unsqueeze(1) - c.unsqueeze(2))
        return (torch.mul(truth_matrix, exp_matrix)).sum(dim=(1, 2))

def compute_loss(model, criterion, logits, target):
    """
        CE + regularization(w.r.t. dissimiarity)
    """
    
    loss = criterion(logits, target.view(-1).long())

    regularity = 0
    stages = 0
    for key in model.out_dict.keys():
        if key.startswith("L_penalty"):
            # accumulate regularization term
            regularity += model.out_dict[key]
            stages += 1
    loss += (regularity / stages)

    return loss