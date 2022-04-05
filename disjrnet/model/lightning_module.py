from functools import partial
import os
import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
import pandas as pd


class LitClassifier(pl.LightningModule):
    def __init__(self, model, learning_rate, class_weight=None, metrics_callbacks=None):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.class_weight = class_weight
        self.metrics_callbacks = metrics_callbacks

        # Important: This property activates manual optimization
        self.automatic_optimization = False

        self.decomp_enabled = self.model.__class__.__name__ == "DisJRNet"

    def log_performance(self, preds, target, phase):
        performances = {}
        for key, func in self.metrics_callbacks.items():
            if key == "auc" and phase == "train":
                continue
            performances[f"{phase}_{key}"] = func(
                preds.view(-1), target).item()

        self.log_dict(performances)
        return performances

    def get_current_fold(self, phase="train"):
        return getattr(self, phase + "_dataloader").dataloader.fold

    def forward(self, inputs):
        return self.model(inputs)

    def parse_batch(self, batch):
        y = batch.pop("label")
        inputs = tuple(batch.values())
        if len(inputs) == 1:
            inputs = inputs[0]  # single input case
        return inputs, y

    def loss_function(self, logits, target, label_smoothing=None, out_dict=None):

        if label_smoothing is not None:
            target = target.float() * (1-label_smoothing) + 0.5 * label_smoothing
            
        cls_weight = getattr(self, 'class_weight', None)
        if cls_weight is not None:
            cls_weight = torch.tensor(cls_weight).to(logits.device)

        if self.model.num_classes == 2:
            pos_weight = cls_weight[1] # pos_weight = 2.0
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            logits = logits.view(-1)
            target = target.float()
            pred = logits.sigmoid()
        else:
            criterion = nn.CrossEntropyLoss(weight=cls_weight)
            pred = logits.argmax(1)
        loss = criterion(logits, target)
        if out_dict is not None:
            regularity = 0
            stages = 0
            for key in out_dict.keys():
                if key.startswith("L_penalty"):
                    # accumulate regularization term
                    regularity += out_dict[key]
                    stages += 1
            loss += (regularity / stages)

        return loss, pred

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        inputs, y = self.parse_batch(batch)
        if self.decomp_enabled:
            logits = self.forward(inputs)
            loss_fn = partial(self.loss_function, out_dict=self.model.out_dict)
        else:
            logits = self.forward(inputs)
            loss_fn = self.loss_function

        loss, pred = loss_fn(logits, y)

        self.manual_backward(loss)
        opt.step()
        opt.zero_grad()

        if self.decomp_enabled:
            params_at_gate = [
                p for p in self.model.parameters() if getattr(p, 'is_gate', False)]
            for p in params_at_gate:
                p.data.clamp_(min=0, max=1)
        self.log("train_loss", loss)
        self.log_performance(pred, y, "train")

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, y = self.parse_batch(batch)
        if self.decomp_enabled:
            logits = self.forward(inputs)
            loss_fn = partial(self.loss_function, out_dict=self.model.out_dict)
        else:
            logits = self.forward(inputs)
            loss_fn = self.loss_function

        loss, pred = loss_fn(logits, y)
        self.log("val_loss", loss)

        return {'val_loss': loss, 'y': y, "y_hat": pred}

    def validation_epoch_end(self, outputs):
        y = torch.cat([x["y"] for x in outputs])
        y_hat = torch.cat([x["y_hat"] for x in outputs])
        if y.float().mean() > 0:
            self.log_performance(y_hat, y, "val")

    def test_step(self, batch, batch_idx):
        inputs, y = self.parse_batch(batch)
        if self.decomp_enabled:
            logits = self.forward(inputs)
            loss_fn = partial(self.loss_function, out_dict=self.model.out_dict)
        else:
            logits = self.forward(inputs)
            loss_fn = self.loss_function

        loss, pred = loss_fn(logits, y)

        self.log("test_loss", loss)
        return {'y': y, "y_hat": pred}

    def test_epoch_end(self, outputs):
        y = torch.cat([x["y"] for x in outputs])
        y_hat = torch.cat([x["y_hat"] for x in outputs])

        # save test results in .csv
        df_test = pd.DataFrame()
        df_test["clip_id"] = self.test_dataloader.dataloader.dataset.clip_list
        df_test["target"] = y.tolist()
        df_test["probs"] = y_hat.view(-1).tolist()
        df_test["preds"] = y_hat.view(-1).ge(0.5).long().tolist()
        submission_dir = os.path.dirname(
            self.trainer.checkpoint_callback.best_model_path)
        submission_path = os.path.join(submission_dir, "testResult.csv")
        df_test.to_csv(submission_path, index=False)
        res = self.log_performance(y_hat, y, "test")

        return res

    def configure_optimizers(self):
        lr = self.learning_rate
        # print(f"current lr : {lr}")
        params = [{'params': [p for p in self.parameters() if not getattr(p, 'is_gate', False)]},
                  {'params': [p for p in self.parameters() if getattr(p, 'is_gate', False)],
                   'weight_decay': 0}]

        optimizer = torch.optim.AdamW(
            params, lr=lr, weight_decay=1e-4)

        return optimizer
