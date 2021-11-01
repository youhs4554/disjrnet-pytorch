import torch
import torch.nn as nn
import torchvision.models.video
from torch.nn.parameter import Parameter
import collections


class SE_Block(nn.Module):
    """
        Squeeze-and-Excitation block
    """

    def __init__(self, num_features, reduction_ratio=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.excitation = nn.Sequential(
            nn.Linear(num_features, num_features //
                      reduction_ratio, bias=False),
            nn.ReLU(True),
            nn.Linear(num_features // reduction_ratio,
                      num_features, bias=False),
            nn.Sigmoid()
        )

        nn.init.kaiming_normal_(
            self.excitation[0].weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(
            self.excitation[2].weight, mode="fan_out", nonlinearity="sigmoid")

    def forward(self, x):

        out = self.squeeze(x)
        out = out.view(out.size(0), -1)
        out = self.excitation(out)
        out = out.view(out.size(0), out.size(1), 1, 1, 1)

        # scale (re-calibration)
        return out * x


class FnBDec(nn.Module):
    # FnBDec: Foreground-and-Background Decomposer
    def __init__(self, num_features, fusion_method="gating"):
        super().__init__()

        # decomposer (use SE_Block to access global information)
        self.decomposer = nn.Sequential(
            SE_Block(num_features, reduction_ratio=16),
            nn.Conv3d(num_features, num_features, kernel_size=1),
            nn.ReLU(True),
        )

        if fusion_method == "gating":
            self.gate = Parameter(torch.Tensor(1, num_features, 1, 1, 1),
                                  requires_grad=True)  # learnable gate
            self.gate.data.fill_(0.5)
            setattr(self.gate, 'is_gate', True)

            # affine layer after gating
            self.affine = nn.Sequential(
                nn.Conv3d(num_features, num_features, kernel_size=1),
                nn.ReLU(True),
            )

        elif fusion_method == "gconv":
            num_groups = 2

            self.gconv = nn.Sequential(
                nn.Conv3d(num_groups * num_features, num_features,
                          kernel_size=3, padding=1, groups=num_groups),
                nn.ReLU(True),
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(
                m.weight, mode="fan_out", nonlinearity="relu")
            # if isinstance(m, nn.Conv3d) and m.bias is not None:
            #     nn.init.constant_(m.bias, 0)

    def fusion(self, x_fg, x_bg):
        if hasattr(self, "gate"):
            # apply learnable gate
            x_hat = self.gate * x_fg + (1-self.gate) * x_bg
            out = self.affine(x_hat)
        else:
            def channel_shuffle(x, num_groups):
                n, c, *dhw = x.shape
                x = x.view(n, num_groups, c//num_groups,
                           *dhw).transpose(1, 2)
                x = x.reshape(n, -1, *dhw)
                return x

            # concat the two component w.r.t channel axis
            x = torch.cat((x_fg, x_bg), dim=1)
            # channel shuffling
            x_shuffled = channel_shuffle(x, num_groups=2)
            # grouped conv
            out = self.gconv(x_shuffled)

        return out

    def forward(self, x):
        # hypothesis : x = x_fg + x_bg
        x_fg = self.decomposer(x)
        x_bg = torch.abs(x - x_fg)

        out = self.fusion(x_fg, x_bg)

        return out + x


class FnBNet(nn.Module):
    def __init__(self, num_class,
                 base_model="r2plus1d_18",
                 dropout=0.8):

        super().__init__()

        self.num_class = num_class

        self._build_base_model(base_model)

        names = [name for name, _ in self.base_model.named_children()]

        self.s1 = getattr(self.base_model, names[0])
        self.s1_fnb = FnBDec(num_features=self.layers[0])

        self.s2 = getattr(self.base_model, names[1])
        self.s2_fnb = FnBDec(num_features=self.layers[1])

        self.s3 = getattr(self.base_model, names[2])
        self.s3_fnb = FnBDec(num_features=self.layers[2])

        self.s4 = getattr(self.base_model, names[3])
        self.s4_fnb = FnBDec(num_features=self.layers[3])

        self.s5 = getattr(self.base_model, names[4])
        self.s5_fnb = FnBDec(num_features=self.layers[4])

        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        if num_class == 2:
            num_outputs = 1
        else:
            num_outputs = num_class
            raise NotImplementedError(
                "Multi-class is not implemnted yet. comming soon!")

        if dropout > 0:
            self.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.layers[4], num_outputs)
            )
        else:
            self.fc = nn.Linear(self.layers[4], num_outputs)

        self.spread_div = collections.OrderedDict()
        self.fhooks = []

        for l in self._modules.keys():
            child = getattr(self, l)
            if hasattr(child, "decomposer"):
                # forward hook for decomposer
                self.fhooks.append(
                    getattr(child, "decomposer").register_forward_hook(self.fwd_hook(l)))

    #     self.fc.apply(self._init_fc)

    # def _init_fc(self, m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.normal_(m.weight, std=0.01)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)

    def fwd_hook(self, layer_name):
        def discrepancy_hook(module, input, output):
            x = input[0]
            x_fg = output
            x_bg = torch.abs(x - x_fg)

            x_fg_mu, x_fg_sd = x_fg.flatten(1).mean(1), x_fg.flatten(1).std(1)
            x_bg_mu, x_bg_sd = x_bg.flatten(1).mean(1), x_bg.flatten(1).std(1)

            fg_dist = torch.distributions.Normal(x_fg_mu, x_fg_sd)
            bg_dist = torch.distributions.Normal(x_bg_mu, x_bg_sd)

            kldiv = torch.distributions.kl_divergence(
                fg_dist, bg_dist).mean().clamp(0, 1)

            inverse_kldiv = 1.0 - kldiv
            # sd_of_bg = x_bg.flatten(1).std(1).mean()
            self.spread_div[layer_name] = inverse_kldiv

        def temporal_voltility_hook(module, input, output):
            x = input[0]
            x_fg = output
            x_bg = torch.abs(x - x_fg)

            x_bg_time_major = x_bg.permute(0, 2, 1, 3, 4)  # (N,T,C,H,W)

            bg_voltility = torch.abs(
                x_bg_time_major[:, 1:]-x_bg_time_major[:, :-1]).sum(dim=(1, 2, 3, 4)).mean()
            # bg_voltility = x_bg_time_major.std(1).sum(dim=(1, 2, 3)).mean()
            # minimize bg's voltility
            self.spread_div[layer_name] = bg_voltility

        return discrepancy_hook

    def _build_base_model(self, base_model):
        # TODO. test more base models (including 2D CNNs)
        # which are listed in https://github.com/facebookresearch/pytorchvideo/blob/main/docs/source/model_zoo.md
        if base_model == "r2plus1d_18" or base_model == "r3d_18":
            self.base_model = getattr(
                torchvision.models.video, base_model)(pretrained=True)
            self.layers = [64, 64, 128, 256, 512]
        elif base_model == "r2plus1d_34":
            self.base_model = torch.hub.load(
                "moabitcoin/ig65m-pytorch",
                "r2plus1d_34_8_ig65m",
                num_classes=487,
                pretrained=True,
            )
            self.layers = [64, 64, 128, 256, 512]

    def forward(self, x):
        x = self.s1(x)
        x = self.s1_fnb(x)
        x = self.s2(x)
        x = self.s2_fnb(x)
        x = self.s3(x)
        x = self.s3_fnb(x)
        x = self.s4(x)
        x = self.s4_fnb(x)
        x = self.s5(x)
        x = self.s5_fnb(x)
        x = self.avg_pool(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x, self.spread_div


def Baseline(num_classes, base_model="r2plus1d_18", dropout=0.0):
    if base_model == "r2plus1d_18" or base_model == "r3d_18":
        model = getattr(
            torchvision.models.video, base_model)(pretrained=True)
        num_features = getattr(model.fc, "in_features")
        num_outputs = num_classes if num_classes > 2 else 1
        if dropout > 0.0:
            model.fc = nn.Sequential(nn.Dropout(
                dropout), nn.Linear(num_features, num_outputs))
        else:
            model.fc = nn.Linear(num_features, num_outputs)

    return model


def test_inference():
    from torchsummary import summary

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = FnBNet(num_class=2)
    model.to(device)
    inputs = torch.randn(4, 3,  10, 112, 112).to(device)
    out = model(inputs)
    input_shape = inputs.shape
    print("\r Input shape(s) : {}, Output shape(s) : {}".format(
        input_shape, out.shape))

    if device.startswith("cuda"):
        device = "cuda"

    print(summary(model, input_shape[1:], device=device))
