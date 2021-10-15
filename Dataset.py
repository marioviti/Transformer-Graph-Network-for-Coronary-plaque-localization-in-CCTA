import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

def sample_patches(volume, coordinates, shape):
    patches = np.concatenate( [ sample_patch(volume, center, shape)[np.newaxis] for center in coordinates ], axis=0)
    return patches

def sample_patch(volume, center, shape):
    begx,begy,begz = np.maximum(0, np.array(center) - np.array(shape)//2. ).round().astype(int)
    endx,endy,endz = np.minimum(np.array(volume.shape), np.array(center) + np.array(shape)/2. ).round().astype(int)
    patch = volume[begx:endx, begy:endy, begz:endz ]
    dx,dy,dz = (np.array(shape) - ( np.array([endx,endy,endz]) - np.array([begx,begy,begz]) )  ).astype(int)
    patch = np.pad(patch, [ (dx//2,dx//2+dx%2), (dy//2,dy//2+dy%2), (dz//2,dz//2+dz%2) ], mode='reflect')
    return patch


def get_k_neigh(node_i, adj, k=3):
  hops, neigh = [],[]
  adjlist = [node_i]
  for k_i in range(1,k+1):
    new_adjlist = []
    for node in adjlist:
      succs = np.arghwhere(adj[node,:]==1).tolist()
      for succ in succs: 
        if succ not in neigh: 
          neig += [succ]
          hops += [k_i]
          new_adjlist += [succ]
      preccs = np.arghwhere(adj[node,:]==-1).tolist()
      for prec in preccs: 
        if prec not in neigh: 
          neig += [prec]
          hops += [-k_i]
          new_adjlist += [prec]
      adjlist = new_adjlist
  hops = [0] + hops
  neigh = [node_i] + neigh
  return hops, neigh


class LocalizationDataset(Dataset):
  
  def __init__(self, data):
    ctscan_folder = os.path.join(data,'ctscan')
    adj_folder = os.path.join(data,'adj')
    coords_folder = os.path.join(data,'coords')
    targets_folder = os.path.join(data,'targets')
    
    ctscan_file_paths = [ os.path.join(ctscan_folder, f) for f in os.listdir(ctscan_folder) ]
    adj_file_paths = [ os.path.join(adj_folder, f) for f in os.listdir(adj_folder) ]
    coords_file_paths = [ os.path.join(coords_file_path, f) for f in os.listdir(coords_file_path) ]
    targets_file_paths = [ os.path.join(targets_file_path, f) for f in os.listdir(targets_file_path) ]

    self.adj_file_paths = adj_file_paths
    self.coords_file_paths = coords_file_paths
    self.targets_file_paths = targets_file_paths
    self.ctscan_file_paths = ctscan_file_paths
  
  def set_data(self):
    self.neighs = []
    self.coords = []
    self.cid = []
    self.targets = []
    self.hops = []
    self.ctscans = []
    
    for cid,(ctscan_file_path, adj_file_path, coords_file_path, targets_file_path) in enumerate(zip(self.ctscan_file_paths, self.adj_file_paths,
                                                                                    self.coords_file_paths, self.targets_file_paths)):
      ctscan = np.load(ctscan_file_path)
      adj = np.load(adj_file_path)
      coords = np.load(coords_file_path)
      targets = np.load(targets_file_path)
      self.ctscans += [ctscan]
      
      N,_ = coords.shape
      
      for i in range(N):
        hops, neighs = get_k_neigh(i,adj,k=7)
        self.hops += [(0,*hops)]
        self.neighs += [neighs]
        self.cid += [cid]
        self.coords += [coords[i]]
        self.targets += [targets[i]]
  
  sampler_w = np.array(self.targets)
  Npos = sampler_w.sum()
  Nneg = (1-sampler_w).sum()
  sampler_w[sampler_w==1] = Nneg
  sampler_w[sampler_w==0] = Npos
  self.sampler_w = sampler_w
  
  def __len__(self): return len(self.targets)
  
  def __getitem__(self,idx):
    ctscan = self.ctscans[self.cid[idx]]
    neighs = self.neighs[idx]
    targets = self.targets[neighs]
    hops = self.hops[neighs]
    coords = self.coords[neighs]
    patches = sample_patches(ctscan, coords, [25,25,25])
    return patches, hops, targets
    
      
    
    
    
    
    
    
