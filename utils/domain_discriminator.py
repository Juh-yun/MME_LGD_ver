from typing import List, Dict
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

# class ReverseLayerF(Function):
#     @staticmethod
#     def forward(ctx, x, alpha):
#         ctx.alpha = alpha
#         return x.view_as(x)
#
#     @staticmethod
#     def backward(ctx, grad_output):        # 오차 역전파 할 때에는 기울기에 -를 붙여서
#         output = grad_output.neg() * ctx.alpha
#         return output, None
#
# class DomainDiscriminator(nn.Module):
#     def __init__(self):
#             super(DomainDiscriminator, self).__init__()
#             self.in_feature = 512
#             self.hidden_size = 1024
#             self.linear1(self.in_feature, self.hidden_size)
#             self.BN1 = nn.BatchNorm1d(self.hidden_size)
#             self.relu = nn.ReLU(True)
#             self.linear2(self.hidden_size, 1)
#
#     def forward(self, x, in_feature, hidden_size, alpha):
#         self.in_feature = in_feature
#         self.hidden_size = hidden_size
#         x = ReverseLayerF.apply(x, alpha)
#         x = self.linear1(x)
#         x = self.BN1(x)
#         x = self.relu(x)
#         self.linear2(x)
#
#         x = F.sigmoid(x)
#         return x
#
class DomainDiscriminator(nn.Sequential):
    def __init__(self, in_feature: int, hidden_size: int, batch_norm=True):
        if batch_norm:
            super(DomainDiscriminator, self).__init__(
                nn.Linear(in_feature, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            )
        else:
            super(DomainDiscriminator, self).__init__(
                nn.Linear(in_feature, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            )

    def get_parameters(self) -> List[Dict]:
        return [{"params": self.parameters(), "lr": 1.}]



class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
#

class GradientReverseLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super(GradientReverseLayer, self).__init__()
        self.alpha = alpha

    def forward(self, input, alpha):
        return ReverseLayerF.apply(input, alpha)


class DomainAdversarialLoss(nn.Module):

    def __init__(self, domain_discriminator: nn.Module):
        super(DomainAdversarialLoss, self).__init__()
        self.grl = GradientReverseLayer()
        self.domain_discriminator = domain_discriminator

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor, alpha, ns):
        f = self.grl(torch.cat((f_s, f_t), dim=0), alpha)
        d = self.domain_discriminator(f)
        d_s = d[:ns]
        d_t = d[ns:]
        d_label_s = torch.ones((f_s.size(0), 1)).to(f_s.device)
        d_label_t = torch.zeros((f_t.size(0), 1)).to(f_t.device)

        return d_s, d_t, d_label_s, d_label_t

