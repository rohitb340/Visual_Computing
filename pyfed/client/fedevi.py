import torch
from .base import BaseClient
from pyfed.manager.helper.build_loss import build_loss
from pyfed.loss.loss import *


class FedEviClient(BaseClient):
    def __init__(self, config, site, server_model, partition=None):
        super(FedEviClient, self).__init__(config, site, server_model, partition)

    def train(self, server_model=None):
        self.loss_fn = build_loss(self.config)
        self.model.to(self.device)
        self.model.train()
        loss_all = 0

        outputs = torch.tensor([], dtype=torch.float32, device=self.device)
        labels = torch.tensor([], dtype=torch.float32, device=self.device)

        for step, (image, label) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            image, label = image.to(self.device), label.to(self.device)
            output = self.model(image)
            loss = self.loss_fn(output, label, self.round+1)
            loss_all += loss.item()
            outputs = torch.cat([outputs, output.detach()], dim=0)
            labels = torch.cat([labels, label.detach()], dim=0)
            loss.backward()
            self.optimizer.step()
            self.curr_iter += 1
        
        loss = loss_all / len(self.train_loader)

        acc = self.metric_fn(outputs, labels)
        self.round += 1

        self.model.to('cpu')
        return loss, acc
    
    @torch.no_grad()
    def val(self, model=None):
        self.loss_fn = DiceLossFundus()
        return super().val(model)
    
    @torch.no_grad()
    def test(self, model=None):
        self.loss_fn = DiceLossFundus()
        return super().test(model)

    def client_to_server(self):
        return {'model': self.model,
                'valid_loader': self.valid_loader,
                'optimizer': self.optimizer,
                'data_len': len(self.train_loader),
                }