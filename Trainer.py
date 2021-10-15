
import os
import torch as th
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import copy

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
