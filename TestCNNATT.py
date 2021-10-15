
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import argparse 

from .Dataset import LocalizationDataset
from .Trainer import TrainCNNATT, create_trainer

import torch as th

def load_lightning_module(checkpoint_path, model_class):
    ckpt = th.load(checkpoint_path)
    pretrained_dict = ckpt['state_dict']
    params = ckpt['hyper_parameters']
    model = model_class(**params)
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict, strict=False)
    return model

if __name__=='__main__':
  
  parser = argparse.ArgumentParser()
  parser.add_argument('--test_data_path', default='test_data')
  parser.add_argument('--ckptfile_path')

  args = parser.parse_args()
  test_data_path = args.__dict__['train_data_path']
  ckptfile_path = args.__dict__['ckptfile_path']
  
  test_data_path = LocalizationDataset(test_data_path)
  test_dataloader = Dataloader(test_data_path,batch_size=batch_size, shuffle=False)
  model = load_lightning_module(ckptfile_path, TrainCNNATT)
  
  outputs = []
  with th.no_grad():
    for batch_idx, batch in enumerate(test_dataloader):
      output = model.test_step(batch, batch_idx)
      outputs += [output]
      
  print('done testing')
