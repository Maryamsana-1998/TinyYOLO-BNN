import torch
import numpy as np


net= torch.load('pretrained.pt', map_location = 'cpu')['network']
i=0
d={}
for key in net:
  if "convbatch.layers.0.weight" in key:
   d['arr_'+ str(i)]=net[key]
   print(key)
   print('arr_'+str(i))
   i=i+1
   d['arr_'+str(i)]=np.zeros(net[key].shape[0],dtype=np.float32)
   print('np.zeros'+ key)
   print('arr_'+str(i))
   i=i+1

  if "_convbatch.layers.1.bias" in key or "_convbatch.layers.1.weight" in key or "convbatch.layers.1.running_mean" in key:
   d['arr_'+str(i)]=net[key]
   print(key)
   print('arr_'+str(i))
   i=i+1

  if "convbatch.layers.1.running_var" in key:
   d['arr_'+str(i)]=1./(np.sqrt(net[key]))
   print(key)
   print('arr_'+str(i))
   i=i+1
  if "_conv.weight" in key:
   d['arr_'+str(i)]=net[key]
   print(key)
   print('arr_'+str(i))
   i=i+1
   d['arr_'+str(i)]=np.transpose(net[key])
   print(key + 'transpose')
   print('arr_'+str(i))
   i=i+1
   d['arr_'+str(i)]=np.zeros(net[key].shape[0],dtype=np.float32)+1e-7
   print(key + 'zeros')
   print('arr_'+str(i))
   i=i+1
   d['arr_'+str(i)]=np.zeros(net[key].shape[0],dtype=np.float32)+1e-7
   print(key + 'zeros')
   print('arr_'+str(i))
   i=i+1
   d['arr_'+str(i)]=np.zeros(net[key].shape[0],dtype=np.float32)+1e-7
   print(key + 'zeros')
   print('arr_'+str(i))
   i=i+1
   d['arr_'+str(i)]=np.zeros(net[key].shape[0],dtype=np.float32)+1e-7
   print(key + 'zeros')
   print('arr_'+str(i))
   i=i+1

np.savez_compressed("modelnew.npz",**d)