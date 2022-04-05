import torch
import torch.nn as nn
import torchvision.models.video
from torch.nn.parameter import Parameter
import collections
from torchvision.models.feature_extraction import create_feature_extractor

class SE_Block(nn.Module):
    """
        Squeeze-and-Excitation block
    """

    def __init__(self, num_features, reduction_ratio=16, dimension=2):
        super().__init__()
        self.squeeze = getattr(nn, "AdaptiveAvgPool" + str(dimension) + "d")(1)
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
        out = out.view(out.size(0), out.size(1), *[1]*(x.dim()-2))

        # scale (re-calibration)
        return out * x


class NonlinearDecomposer(nn.Module):
    def __init__(self, num_features, dimension=2):
        super().__init__()
        
        conv_builder = getattr(nn, "Conv"+str(dimension)+"d")

        # decomposer (use SE_Block to access global information)
        self.decomposer = nn.Sequential(
            SE_Block(num_features, reduction_ratio=16, dimension=dimension),
            conv_builder(num_features, num_features, kernel_size=1),
            nn.ReLU(True),
        )

    def forward(self, x):
        x_fg = self.decomposer(x)
        x_bg = torch.abs(x - x_fg)

        return x_fg, x_bg


class DisJR_Module(nn.Module):
    # DisJR_Module: Module for Disjointing Representation 
    def __init__(self, num_features, fusion_method="gating", dimension=2):
        super().__init__()
        
        conv_builder = getattr(nn, "Conv"+str(dimension)+"d")

        self.decomposer = NonlinearDecomposer(num_features, dimension=dimension)

        if fusion_method == "gating":
            self.gate = Parameter(torch.Tensor(1, num_features, *[1]*dimension),
                                  requires_grad=True)  # learnable gate
            self.gate.data.fill_(0.5)
            setattr(self.gate, 'is_gate', True)

            # affine layer after gating
            self.affine = nn.Sequential(
                conv_builder(num_features, num_features, kernel_size=1),
                nn.ReLU(True),
            )

        elif fusion_method == "gconv":
            self.num_groups = 2

            self.gconv = nn.Sequential(
                conv_builder(self.num_groups * num_features, num_features,
                          kernel_size=3, padding=1, groups=num_features),
                nn.ReLU(True),
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if "Conv" in m.__class__.__name__:
            nn.init.kaiming_normal_(
                m.weight, mode="fan_out", nonlinearity="relu")

    def fusion(self, x_fg, x_bg):
        if hasattr(self, "gate"):
            # apply learnable gate
            x_hat = self.gate * x_fg + (1-self.gate) * x_bg
            out = self.affine(x_hat)
        else:
            def channel_shuffle(x):
                n, c, *sizes = x.shape
                x = x.view(n, self.num_groups, c//self.num_groups,
                           *sizes).transpose(1, 2)
                x = x.reshape(n, -1, *sizes)
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


class DisJRNet(nn.Module):
    def __init__(self, num_classes,
                 base_model="r2plus1d_18", dimension=2,
                 dropout=0.8, margin=0.0, fusion_method="gating"):

        super().__init__()

        self.num_classes = num_classes
        self.dimension = dimension
        self.margin = margin

        self._build_base_model(base_model, dimension)

        names = self.names

        self.s1_disjr = DisJR_Module(
            num_features=self.layers[0], fusion_method=fusion_method, dimension=dimension)

        self.s2 = getattr(self.base_model, names[1])
        self.s2_disjr = DisJR_Module(
            num_features=self.layers[1], fusion_method=fusion_method, dimension=dimension)

        self.s3 = getattr(self.base_model, names[2])
        self.s3_disjr = DisJR_Module(
            num_features=self.layers[2], fusion_method=fusion_method, dimension=dimension)

        self.s4 = getattr(self.base_model, names[3])
        self.s4_disjr = DisJR_Module(
            num_features=self.layers[3], fusion_method=fusion_method, dimension=dimension)

        self.s5 = getattr(self.base_model, names[4])
        self.s5_disjr = DisJR_Module(
            num_features=self.layers[4], fusion_method=fusion_method, dimension=dimension)

        self.avg_pool = getattr(nn, "AdaptiveAvgPool" + str(dimension) + "d")(1)

        if num_classes == 2:
            num_outputs = 1
        else:
            num_outputs = num_classes

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
            postfix = "_disjr"
            self.out_dict["fusion_activation_{}".format(
                layer_name[:-len(postfix)])] = output
        return intermediate_fusion_hook

    def component_activation_hook(self, layer_name):
        def penalty_hook(module, input, output):
            x = input[0]

            x_fg, x_bg = output

            # channel descriptor vectors for each component
            x_fg_vec = x_fg.flatten(2).mean(2)
            x_bg_vec = x_bg.flatten(2).mean(2)

            # channel statistics for each component
            x_fg_mu, x_fg_sd = x_fg_vec.mean(1), x_fg_vec.std(1)
            x_bg_mu, x_bg_sd = x_bg_vec.mean(1), x_bg_vec.std(1)

            fg_dist = torch.distributions.Normal(x_fg_mu, x_fg_sd)
            bg_dist = torch.distributions.Normal(x_bg_mu, x_bg_sd)

            kldiv = torch.distributions.kl_divergence(
                fg_dist, bg_dist).mean()

            L_penalty = torch.maximum(
                self.margin - kldiv, torch.tensor(0.0).to(x.device))

            postfix = "_disjr"
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

    def _build_base_model(self, base_model, dimension):
        if dimension == 3:
            if base_model == "r2plus1d_18" or base_model == "r3d_18":
                self.base_model = getattr(
                    torchvision.models.video, base_model)(pretrained=True)
            elif base_model == "r2plus1d_34":
                self.base_model = torch.hub.load(
                    "moabitcoin/ig65m-pytorch",
                    "r2plus1d_34_8_ig65m",
                    num_classes=487,
                    pretrained=True,
                )
            dummy_ = torch.randn(1,3,10,224,224)
            stem_node = {'stem': 's1'}
            return_nodes = {'stem': 's1',
                            'layer1' : 's2',
                            'layer2' : 's3',
                            'layer3' : 's4',
                            'layer4' : 's5'}
        elif dimension == 2:
            assert base_model.startswith('resnet'), f"Do not support model name of '{base_model}'. ResNet-based model is only supported."
            self.base_model = getattr(
                torchvision.models, base_model)(pretrained=True)
            dummy_ = torch.randn(1,3,224,224)
            return_nodes = {'maxpool': 's1',
                            'layer1' : 's2',
                            'layer2' : 's3',
                            'layer3' : 's4',
                            'layer4' : 's5'}
            stem_node = {'maxpool': 's1'}
            
        self.stem_extractor = create_feature_extractor(self.base_model, stem_node)
        feats = create_feature_extractor(self.base_model, return_nodes)(dummy_)
        self.layers = [ t.size(1) for t in feats.values() ]
        self.names = list(return_nodes.keys())

    def forward(self, x):
        
        feats = self.stem_extractor(x)
        x = self.s1_disjr(feats['s1'])
        x = self.s2(x)
        x = self.s2_disjr(x)
        x = self.s3(x)
        x = self.s3_disjr(x)
        x = self.s4(x)
        x = self.s4_disjr(x)
        x = self.s5(x)
        x = self.s5_disjr(x)
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
