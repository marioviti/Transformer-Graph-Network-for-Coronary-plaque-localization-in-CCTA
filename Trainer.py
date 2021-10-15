import os
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import copy

from .Models import CNNATT, WindowingHU, Augmentation

class TrainCNNATT(pl.LightningModule):
    
    def __init__(self, hparams, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.cnn = CNNATT(input_channels=1, output_channels=1)
        self.preprocess = WindowingHU(**preprocess_hparams)
        self.augmentation = Augmentation()
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()

    def forward(self, x, pos):
        y = self.cnn(self.preprocess(x), pos)
        return y
    
    def get_log(self,outputs):
        loc_targets = th.cat([ x['targets'] for x in outputs] ).squeeze()
        loc_logits = th.cat([ x['logits'] for x in outputs] ).squeeze()
        loss = th.tensor([ x['loss'] for x in outputs]).mean()
        log['loss'] = loss
        return log
    
    def training_epoch_end(self, outputs):
        log = self.get_log(outputs)
        self.trainer.train_log(log)
        
    def validation_epoch_end(self, outputs):
        log = self.get_log(outputs)
        self.trainer.valid_log(log)
        log = { 'val_loss' : log['loss'], **log }
        return log
    
    def training_step(self, batch, batch_idx):
        x, pos, targets = batch
        batch_size, neighbourhood, input_channels, depth, height, width = x.shape
        x = x.view(batch_size*neighbourhood, input_channels, depth, height, width)
        x = self.augmentation(x)
        _ , input_channels, depth, height, width = x.shape
        x = x.view(batch_size, neighbourhood, input_channels, depth, height, width)
        logits = self(x,pos)
        loss = self.BCEWithLogitsLoss(logits[targets>=0].squeeze(), targets[targets>=0].squeeze())
        logits = logits[:,0,:]
        targets = targets[:,0]
        output = { 'logits': logits, 'targets': targets, 'loss': loss }
        return output

    def validation_step(self, batch, batch_idx):
        x, pos, targets = batch
        logits = self(x,pos)
        logits = logits[:,0,:]
        targets = targets[:,0]
        loss = self.BCEWithLogitsLoss(logits.squeeze(), targets.squeeze())
        output = { 'logits': logits, 'targets': targets, 'loss': loss }
        return output
    
    def configure_optimizers(self):
        optimizer0 = th.optim.Adam(self.cnn.parameters(), lr=0.0001)
        return { 'optimizer': optimizer0 }

class TrainEvalTrainer(pl.Trainer):
    def __init__(self, *args,  train_logger=None, valid_logger=None, **kwargs ):
        self.train_logger = train_logger
        self.valid_logger = valid_logger
        self.validation_log = {}
        pl.Trainer.__init__(self, *args, **kwargs)
        
    def is_better_epoch(self, validation_log, best_measure, mode='max'):
        if best_measure in self.validation_log:
            return self.validation_log[best_measure] < validation_log[best_measure]
        return True
        
    def set_best_validation_log(self, validation_log, best_measure, mode='max'):
        self.validation_log['best_measure'] = best_measure
        if self.is_better_epoch(validation_log, best_measure, mode='max'):
            self.validation_log = copy.deepcopy(validation_log)
            self.validation_log['best_epoch'] = self.current_epoch
            
    def train_add_images(self, tag, images):
        self.train_logger.experiment.add_images(tag, images, global_step=self.current_epoch)
        
    def valid_add_images(self, tag, images):
        self.valid_logger.experiment.add_images(tag, images, global_step=self.current_epoch)
            
    def train_log(self, log):
        for k,v in log.items():
            self.train_logger.experiment.add_scalar(k,v,self.current_epoch)
        self.train_logger.experiment.flush()
        
    def valid_log(self, log):
        for k,v in log.items():
            self.valid_logger.experiment.add_scalar(k,v,self.current_epoch)
        self.valid_logger.experiment.flush()
        
def create_trainer(log_path, max_epochs=50, precision=32, auto_lr_find=True, gradient_clip_val=0, save_top_k=-1, check_val_every_n_epoch=1, gpus=1, gpu_choice='auto', limit_train_batches=1.):
    os.makedirs(log_path, exist_ok=True)
    train_logger = pl_loggers.TensorBoardLogger(os.path.join(log_path,'train'))
    valid_logger = pl_loggers.TensorBoardLogger(os.path.join(log_path,'valid'))
    checkpoint_callback = ModelCheckpoint(os.path.join(train_logger.log_dir, 'checkpoints'), save_top_k=save_top_k)
    if precision == 16: print('deprecated. use only full precision (32)')
    trainer = TrainEvalTrainer(precision=precision, max_epochs=max_epochs, gpus=gpus, check_val_every_n_epoch=check_val_every_n_epoch, auto_lr_find=auto_lr_find, 
                                   checkpoint_callback=checkpoint_callback, train_logger=train_logger, valid_logger=valid_logger, 
                                   gradient_clip_val=gradient_clip_val, limit_train_batches=limit_train_batches, logger=False)
    
    return trainer
