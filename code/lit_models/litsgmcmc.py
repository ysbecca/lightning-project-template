

import torch
from torch.optim import Adam, SGD, lr_scheduler
import pytorch_lightning as pl
import numpy as np

# from pytorch_lightning.metrics import Accuracy, PrecisionRecallCurve, ROC
from pytorch_lightning.callbacks import LearningRateMonitor

import torch.nn.functional as F

from metrics import ece_loss
from network_models.alexnet import *
from network_models.vgg import *



from global_config import *


class LitSGMCMC(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        
        self.learning_rate = args.lr
        self.model_code = args.model_code
        self.dataset_code = args.dataset_code
        self.optim = args.optim_code
        self.use_scheduler = args.use_scheduler
        self.patience = args.patience
        self.model_depth_code = args.model_depth_code
        self.model = self.__get_model()

        # print(self.model)
        # exit()
        self.save_hyperparameters()

        self.y_preds = []
        self.y_preds_unc = []


    def __get_model(self):
        if self.model_code == LENET_CODE:
            return LeNet()
        elif self.model_code == VGG_CODE:
            return vgg16(num_classes=NUM_CLASSES[self.dataset_code])
        elif self.model_code == ALEX_CODE:
            return alexnet(num_classes=NUM_CLASSES[self.dataset_code], channels=DIMS[self.dataset_code])
        elif self.model_code == RESNET_CODE:
            return resnet34(num_classes=NUM_CLASSES[self.dataset_code])
        elif self.model_code == CNN_CODE:
            return CNN(num_classes=NUM_CLASSES[self.dataset_code])

    def configure_optimizers(self):
        optimizer = None
        scheduler = None

        if self.optim == ADAM_OPTIM:
            optimizer = Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.001)
        elif self.optim == SGD_OPTIM:
            optimizer = SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        
        if self.use_scheduler:
            scheduler = {
                "scheduler": lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=0.1,
                    patience=self.patience,
                    min_lr=5e-7,
                    # verbose=True,
                ),
                'name': 'learning_rate',
                'interval':'epoch',
                'frequency': 1,
                'monitor': 'val_loss'
            }
            
            return [optimizer], [scheduler]
        else:
            return optimizer
    

    def cross_entropy_loss(self, y_pred, y):
        """ takes y_pred post softmax """

        return F.nll_loss(y_pred, y.long())

    def accuracy(self, y_pred, y):
        acc = (y == torch.argmax(y_pred, dim=1)).sum() / torch.tensor(y.shape[0])
        return acc.item() * 100.0

    def ece(self, y_pred, y):
        ece = ece_loss(y.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
        return ece

    def forward(self, x):
        outputs = self.model(x)

        return outputs      

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.cross_entropy_loss(y_pred, y)
        acc = self.accuracy(y_pred, y)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.cross_entropy_loss(y_pred, y)
        acc = self.accuracy(y_pred, y)
        ece = self.ece(y_pred, y)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_ece', ece, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return {'val_loss': loss}


    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.cross_entropy_loss(y_pred, y)
        acc = self.accuracy(y_pred, y)

        for y_ in y_pred:
            self.y_preds_unc.append(y_.detach().cpu().numpy())


        return {'val_loss': loss, 'val_acc': acc}

    def teardown(self, stage):
        if stage != "fit":
            if self.y_preds_unc != []:
                self.y_preds_unc = np.array(self.y_preds_unc)
                self.y_preds = np.argmax(self.y_preds_unc, axis=1)



