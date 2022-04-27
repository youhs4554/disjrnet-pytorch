# DisJR Networks: Disjointed Representation Learning for Better Fall Recognition

DisJR(**Disj**ointing **R**epresentation) is an effective and simple computational unit that disjoints human from unwanted elements(e.g., background) in the video scene without any hints about the human region.
Our proposed DisJR operation is designed to reflect relations between human and various surrounding contexts from data itself, not preprocessed data.
In contrast to the existing methods that uses preprocessed data for the human region, the proposed DisJR operations do not rely on the fixed region.
Instead, the proposed method learns how to separate representations of human region and unwanted elements through explicit _feature-level_ decomposition, i.e., DisJR.
In this way, the model grasps more general representations about the video scene.

## Model overview

![model_overview](imgs/model.png)

## Examples

Here is code example for using DisJRNet:

```
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from disjrnet.model.models import DisJRNet
from disjrnet.model.loss import compute_loss

alpha           =   2.0       # hyperparameter
fusion_method   =   'gating'   # candidates = 'gating' | 'gconv'

# 2D CNN
model = DisJRNet(num_classes    =   10,
                base_model      =   'resnet50',
                dimension       =   2,
                dropout         =   0.8,
                margin          =   alpha,
                fusion_method   =   fusion_method)

# # 3D CNN
# model = DisJRNet(num_classes    =   10,
#                 base_model      =   'r2plus1d_18',
#                 dimension       =   3,
#                 dropout         =   0.8,
#                 margin          =   alpha,
#                 fusion_method   =   fusion_method)

# classification loss = CE
criterion = nn.CrossEntropyLoss()

# dummy data example
inps = torch.randn(10, 3, 112, 112)
tgts = torch.arange(10, dtype=torch.float32).view(10,-1)

dataset = TensorDataset(inps, tgts)
loader = DataLoader(dataset, batch_size=8)
loader_iter = iter(loader)

inputs, target = next(loader_iter)

logits = model(inputs)

loss = compute_loss(model, criterion, logits, target)
pred = logits.argmax(1)

print(f"loss : {loss:.4f}, pred : {pred}, target : {target.view(-1)}")
```
