import os
import torch
import torchvision
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from model import *
from tqdm import *

import pandas as pd
import umap

import matplotlib.pyplot as plt





class Trainer(object):

    def __init__(self, model, sample_path="sample", recon_path="recon", codes_path="codes", checkpoints='model.cp'):

        self.model = model
        self.sample_path = sample_path
        self.recon_path = recon_path
        self.codes_path = codes_path
        self.checkpoints = checkpoints

        self.start_epoch = 0


    def sample_frames(self, epoch):
        with torch.no_grad():
            x_gen = self.model.gen_img(20, mult=1.).view(20, self.model.in_channels, self.model.in_size, self.model.in_size)
            torchvision.utils.save_image(x_gen,'%s/epoch%d.png' % (self.sample_path,epoch), nrow=5)
    
    def recon_frame(self, epoch, original):
        with torch.no_grad():
            recon = self.model(original)[0]

            original = original.view(-1, self.model.in_channels, self.model.in_size, self.model.in_size)
            original = original[0:10]
            recon = recon.view(-1, self.model.in_channels, self.model.in_size, self.model.in_size)
            recon = recon[0:10]

            image = torch.cat((original,recon),dim=0)
            image = image.view(20, self.model.in_channels, self.model.in_size, self.model.in_size)

            torchvision.utils.save_image(image,'%s/epoch%d.png' % (self.recon_path,epoch), nrow=10)

    def umap_codes(self, epoch):

      codes, _ = self.model.gen_codes(512)

      codes = codes.cpu().detach().numpy()

      embedding = umap.UMAP(n_neighbors=15,
                      min_dist=0.1,
                      metric='correlation').fit_transform(codes)

      plt.figure(figsize=(12,12))
      plt.scatter(embedding[:, 0], embedding[:, 1], 
                  edgecolor='none', 
                  alpha=0.80, 
                  s=10)
      plt.axis('off')

      plt.savefig('%s/epoch%d.png' % (self.codes_path, epoch))
      plt.close()


    def save_checkpoint(self,epoch):
        torch.save({
            'epoch' : epoch+1,
            'state_dict' : self.model.state_dict(),
            },
            self.checkpoints)
        
    def load_checkpoint(self):
        try:
            print("Loading Checkpoint from '{}'".format(self.checkpoints))
            checkpoint = torch.load(self.checkpoints)
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            print("Resuming Training From Epoch {}".format(self.start_epoch))
        except:
            print("No Checkpoint Exists At '{}'.Start Fresh Training".format(self.checkpoints))
            self.start_epoch = 0

    def train_model(self, trainloader, epochs=100):

        self.model.train()

        #avgDiff = 0
        for epoch in range(self.start_epoch, epochs):
           #trainloader.shuffle()
           losses = []
           kld_fs = []
           kld_zs = []
           print("Running Epoch : {}".format(epoch+1))
           print(len(trainloader))


           lastDiff = 0
           #lastDiff = avgDiff
           avgDiff = 0

           for i, dataitem in tqdm(enumerate(trainloader, 1)):
               if i >= len(trainloader):
                break
               data, _ = dataitem
               data = data.cuda()

               TINY = 1e-15

               loss4 = torch.Tensor([0])


               if lastDiff > 0.:
                 self.model.zero_grad_all()

                 x_recon, enc_score = self.model(data)

                 loss1 = 1000 * F.mse_loss(x_recon, data, reduction='mean') - torch.mean(enc_score)

                 #we want enc to confuse discr and have discr give 1 to real data(even though it should give 0)

                 loss1.backward()


                 self.model.enc_dec_step()

               else:

                 self.model.zero_grad_all()

                 x_recon, _ = self.model(data)

                 loss1 = 1000 * F.mse_loss(x_recon, data, reduction='mean')

                 #only recon loss when discr is already confused

                 loss1.backward()


                 self.model.enc_dec_step()


               ##

               self.model.zero_grad_all()

               discr_real_score = self.model.forward_score(data)

               loss2 = torch.mean(discr_real_score)
               #we want discr to give 0 for real data

               if loss2.item() > -0.95:
                   loss2.backward()

                   self.model.discr.optimizer.step()

               ##

               self.model.zero_grad_all()

               _, discr_gen_score = self.model.gen_codes()



               loss3 = -torch.mean(discr_gen_score)
               #we want discr to give 1 for generated data

               if loss3.item() > -0.95:
                   loss3.backward()

                   self.model.discr.optimizer.step()

               ##

               if lastDiff > 0.:

                 self.model.zero_grad_all()

                 _, discr_gen_score = self.model.gen_codes()


                 loss4 = torch.mean(discr_gen_score)
                 #we want distr to get 0 (to be more similar to discr)

                 loss4.backward()

                 self.model.distr.optimizer.step()


               if i % 100 == 0:
                print(loss1, loss2, loss3, loss4)

               total_loss = loss1 + loss2 + loss3 + loss4

               avgDiff += loss3.item() * -1 - loss2.item()
               lastDiff = loss3.item() * -1 - loss2.item()



               #loss.backward()
               #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
               #self.optimizer.step()
               losses.append(total_loss.item())
               #discr_real.append((loss2 + )
               #kld_fs.append(kld_f.item())
               #kld_zs.append(kld_z.item())
           #discr_meanloss = np.mean(discr_loss)
           meanloss = np.mean(losses)

           avgDiff /= len(trainloader)
           #meanf = np.mean(kld_fs)
           #meanz = np.mean(kld_zs)
           #self.epoch_losses.append(meanloss)
           print("Epoch {} : Average Loss: {}".format(epoch+1, meanloss))

           print("Disc. quality: {}".format(avgDiff))
           self.save_checkpoint(epoch)


           self.model.eval()
           _, (sample, _)  = next(enumerate(trainloader))
           sample = torch.unsqueeze(sample,0)
           sample = sample.cuda()
           if 1:#(epoch + 1) % 1 == 0:
            self.sample_frames(epoch+1)
            self.recon_frame(epoch+1,sample)
           self.model.train()
        print("Training is complete")