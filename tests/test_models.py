import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from disjrnet.model.models import DisJRNet

alpha           =   2.0       # hyperparameter
fusion_method   =   'gating'   # candidates = 'gating' | 'gconv'

def compute_loss(model, criterion, logits, target):
    """
        CE + dissimiarity regularization
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

def infer(model, dimension):

    # classification loss = CE
    criterion = nn.CrossEntropyLoss()
    
    # dummy data example    
    if dimension == 2:
        inps = torch.randn(10, 3, 112, 112)
    elif dimension == 3:
        inps = torch.randn(10, 3, 10, 112, 112)
        
    tgts = torch.arange(10, dtype=torch.float32).view(10,-1)

    # dataset & dataloader
    dataset = TensorDataset(inps, tgts)
    loader = DataLoader(dataset, batch_size=8)
    loader_iter = iter(loader)

    inputs, target = next(loader_iter)

    # model infer
    logits = model(inputs)

    # calculate loss
    loss = compute_loss(model, criterion, logits, target)
    pred = logits.argmax(1)

    print(f"loss : {loss:.4f}, pred : {pred}, target : {target.view(-1)}")
    
    return loss, pred
    
def test_2d_infer():
    dimension       =   2

    # 2D CNN
    model = DisJRNet(num_classes    =   10,
                    base_model      =   'resnet50',
                    dimension       =   dimension,
                    dropout         =   0.8,
                    margin          =   alpha,
                    fusion_method   =   fusion_method)
    
    loss, pred = infer(model, dimension)
    assert loss > 0
    
def test_3d_infer():
    dimension       =   3

    # 3D CNN
    model = DisJRNet(num_classes    =   10,
                    base_model      =   'r2plus1d_18',
                    dimension       =   dimension,
                    dropout         =   0.8,
                    margin          =   alpha,
                    fusion_method   =   fusion_method)
    loss, pred = infer(model, dimension)
    assert loss > 0