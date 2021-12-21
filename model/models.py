from turtle import forward
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


class NonlinearDecomposer(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        # decomposer (use SE_Block to access global information)
        self.decomposer = nn.Sequential(
            SE_Block(num_features, reduction_ratio=16),
            nn.Conv3d(num_features, num_features, kernel_size=1),
            nn.ReLU(True),
        )

    def forward(self, x):
        x_fg = self.decomposer(x)
        x_bg = torch.abs(x - x_fg)

        return x_fg, x_bg


class FnBDec(nn.Module):
    # FnBDec: Foreground-and-Background Decomposer
    def __init__(self, num_features, fusion_method="gating"):
        super().__init__()

        self.decomposer = NonlinearDecomposer(num_features)

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
            self.num_groups = 2

            self.gconv = nn.Sequential(
                nn.Conv3d(self.num_groups * num_features, num_features,
                          kernel_size=3, padding=1, groups=num_features),
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
            def channel_shuffle(x):
                n, c, *dhw = x.shape
                x = x.view(n, self.num_groups, c//self.num_groups,
                           *dhw).transpose(1, 2)
                x = x.reshape(n, -1, *dhw)
                return x

            # concat the two component w.r.t channel axis
            x = torch.cat((x_fg, x_bg), dim=1)
            # channel shuffling
            x_shuffled = channel_shuffle(x)
            # grouped conv
            out = self.gconv(x_shuffled)

        return out

    def forward(self, x):
        x_fg, x_bg = self.decomposer(x)
        out = self.fusion(x_fg, x_bg)

        return out + x # residual connect


class FnBNet(nn.Module):
    def __init__(self, num_classes,
                 base_model="r2plus1d_18",
                 dropout=0.8, margin=0.0, fusion_method="gating"):

        super().__init__()

        self.num_classes = num_classes
        self.margin = margin

        self._build_base_model(base_model)

        names = [name for name, _ in self.base_model.named_children()]

        self.s1 = getattr(self.base_model, names[0])
        self.s1_fnb = FnBDec(
            num_features=self.layers[0], fusion_method=fusion_method)

        self.s2 = getattr(self.base_model, names[1])
        self.s2_fnb = FnBDec(
            num_features=self.layers[1], fusion_method=fusion_method)

        self.s3 = getattr(self.base_model, names[2])
        self.s3_fnb = FnBDec(
            num_features=self.layers[2], fusion_method=fusion_method)

        self.s4 = getattr(self.base_model, names[3])
        self.s4_fnb = FnBDec(
            num_features=self.layers[3], fusion_method=fusion_method)

        self.s5 = getattr(self.base_model, names[4])
        self.s5_fnb = FnBDec(
            num_features=self.layers[4], fusion_method=fusion_method)

        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        if num_classes == 2:
            num_outputs = 1
        else:
            num_outputs = num_classes
            raise NotImplementedError(
                "Multi-class is not implemnted yet. comming soon!")

        if dropout > 0:
            self.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.layers[4], num_outputs)
            )
        else:
            self.fc = nn.Linear(self.layers[4], num_outputs)

        self.out_dict = collections.OrderedDict()

        for l in self._modules.keys():
            child = getattr(self, l)
            if hasattr(child, "decomposer"):
                # forward hook for decomposer
                getattr(child, "decomposer").register_forward_hook(
                    self.component_activation_hook(l))
                # forward hook for child module
                child.register_forward_hook(self.fusion_activation_hook(l))

    def fusion_activation_hook(self, layer_name):
        def intermediate_fusion_hook(module, input, output):
            postfix = "_fnb"
            self.out_dict["fusion_activation_{}".format(
                layer_name[:-len(postfix)])] = output
        return intermediate_fusion_hook

    def component_activation_hook(self, layer_name):
        def penalty_hook(module, input, output):
            x = input[0]

            x_fg, x_bg = output

            # channel descriptor vectors for each component
            x_fg_vec = x_fg.mean(dim=(2, 3, 4))
            x_bg_vec = x_bg.mean(dim=(2, 3, 4))

            # channel statistics for each component
            x_fg_mu, x_fg_sd = x_fg_vec.mean(1), x_fg_vec.std(1)
            x_bg_mu, x_bg_sd = x_bg_vec.mean(1), x_bg_vec.std(1)

            fg_dist = torch.distributions.Normal(x_fg_mu, x_fg_sd)
            bg_dist = torch.distributions.Normal(x_bg_mu, x_bg_sd)

            kldiv = torch.distributions.kl_divergence(
                fg_dist, bg_dist).mean()

            L_penalty = torch.maximum(
                self.margin - kldiv, torch.tensor(0.0).to(x.device))

            postfix = "_fnb"
            self.out_dict["L_penalty_{}".format(
                layer_name[:-len(postfix)])] = L_penalty

            self.out_dict["fg_activation_{}".format(
                layer_name[:-len(postfix)])] = x_fg
            self.out_dict["bg_activation_{}".format(
                layer_name[:-len(postfix)])] = x_bg

        def temporal_voltility_hook(module, input, output):
            x = input[0]
            x_fg = output
            x_bg = torch.abs(x - x_fg)

            x_bg_time_major = x_bg.permute(0, 2, 1, 3, 4)  # (N,T,C,H,W)

            bg_voltility = torch.abs(
                x_bg_time_major[:, 1:]-x_bg_time_major[:, :-1]).sum(dim=(1, 2, 3, 4)).mean()
            # bg_voltility = x_bg_time_major.std(1).sum(dim=(1, 2, 3)).mean()
            # minimize bg's voltility
            self.out_dict[layer_name] = bg_voltility

        return penalty_hook

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

        return x


def Baseline(num_classes, base_model="r2plus1d_18", dropout=0.0):
    if base_model == "r2plus1d_18" or base_model == "r3d_18":
        model = getattr(
            torchvision.models.video, base_model)(pretrained=True)
    elif base_model == "r2plus1d_34":
        model = torch.hub.load(
            "moabitcoin/ig65m-pytorch",
            "r2plus1d_34_8_ig65m",
            num_classes=487,
            pretrained=True,
        )

    num_features = getattr(model.fc, "in_features")
    num_outputs = num_classes if num_classes > 2 else 1
    if dropout > 0.0:
        model.fc = nn.Sequential(nn.Dropout(
            dropout), nn.Linear(num_features, num_outputs))
    else:
        model.fc = nn.Linear(num_features, num_outputs)

    return model


def show_model(arch, **kwargs):
    assert isinstance(arch, str)

    from torchsummary import summary

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = eval(arch)(**kwargs)
    model.to(device)
    inputs = torch.randn(4, 3,  10, 112, 112).to(device)
    out = model(inputs)
    input_shape = inputs.shape
    print("\r Input shape(s) : {}, Output shape(s) : {}".format(
        input_shape, out.shape))

    if device.startswith("cuda"):
        device = "cuda"

    print(summary(model, input_shape[1:], device=device))

    return True
