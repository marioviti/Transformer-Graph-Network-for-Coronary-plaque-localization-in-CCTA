from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import argparse 

from .Dataset import LocalizationDataset
from .Trainer import TrainCNNATT, create_trainer

if __name__=='__main__':
  num_epochs = 25
  batch_size = 32
  
  parser = argparse.ArgumentParser()
  parser.add_argument('--train_data_path', default='train_data')
  parser.add_argument('--valid_data_path', default='valid_data')

  args = parser.parse_args()
  train_data_path = args.__dict__['train_data_path']
  valid_data_path = args.__dict__['valid_data_path']
  
  train_dataset = LocalizationDataset(train_data_path)
  valid_dataset = LocalizationDataset(valid_data_path)
  
  sampler = WeightedRandomSampler(train_dataset.sampler_w)
  train_dataloader = Dataloader(train_dataset,batch_size=batch_size, sampler=sampler)
  valid_dataloader = Dataloader(valid_dataset,batch_size=batch_size, shuffle=False)
  
  model = TrainCNNATT()
  trainer = create_trainer('logs/CNNATT', max_epochs=num_epochs)
  trainer.fit(model, train_dataloader=train_dataloader, valid_dataloader=valid_dataloader)
