import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from train import Trainer

from model import PAAE

import matplotlib.pyplot as plt

from data import CelebaDataset


batch_size = 32

model = PAAE(p_dim = 32, in_size=64, in_channels=3)

custom_transform = transforms.Compose([transforms.ToTensor()])

train_dataset = CelebaDataset(txt_path='/home/samuel/Data/CelebAligned/list_attr_celeba.txt',
                              img_dir='/home/samuel/Data/CelebAligned/',
                              transform=custom_transform)

trainloader = DataLoader(dataset=train_dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=1)

examples = enumerate(trainloader)
#plt.show()

trainer = Trainer(model)

trainer.model.eval()
_, (sample, _) = next(examples)
print(sample.shape)
sample = torch.unsqueeze(sample,0)
sample = sample.cuda()

trainer.umap_codes(0)
trainer.sample_frames(0)
trainer.recon_frame(0, sample)

trainer.train_model(trainloader)

