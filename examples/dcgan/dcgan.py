
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

#TODO : apply init function 

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.utils as vutils
from train_loop.utils.visualization import show_tensor
# Set random seed for reproducibility
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True)

# DCGAN hyperparameters
nz = 100  # Size of z latent vector (i.e. size of generator input)
ngf = 64  # Size of feature maps in generator
ndf = 64  # Size of feature maps in discriminator
nc = 3    # Number of channels in the training images

class CelebADataset(torch.utils.data.Dataset):
    """
    CelebA dataset that returns dicts with keys 'inp' and 'gt'.
    """
    def __init__(self, dataroot='data/celeba', image_size=64, train=True):
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.dataset = datasets.ImageFolder(
            root=dataroot,
            transform=transform
        )
        self.collate_fn = None  # Use default collate function

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        inp, gt = self.dataset[idx]
        return {"inp": inp, "gt_class_id": gt}


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, ngpu=1):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

        self.apply(weights_init)


    def forward(self, batch):
        device = batch['inp'].device
        image_batch = batch['inp']
        # Generate random noise
        noise = torch.randn(image_batch.size(0), nz, 1, 1, device=device)
        generated = self.main(noise)
        return {"generated_image": generated}

class Discriminator(nn.Module):
    def __init__(self, ngpu=1):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        self.apply(weights_init)

    def forward(self, input):
        return self.main(input)
    


class GeneratorLoss(torch.nn.Module):
    """
    Generator loss for GAN training.
    The generator tries to fool the discriminator by making fake images 
    be classified as real.
    """
    def __init__(self, discriminator, generated_image_key="generated_image"):
        super(GeneratorLoss, self).__init__()
        self.discriminator = discriminator
        self.generated_image_key = generated_image_key
        self.criterion = nn.BCELoss()
        
    def forward(self, batch, model_outs):
        # Get the generated (fake) images from generator output
        fake_images = model_outs[self.generated_image_key]
        
        # Create real labels - generator wants discriminator to classify fakes as real
        label_real = torch.ones(fake_images.size(0), device=fake_images.device)
        
        # Get discriminator's prediction on fake images
        fake_pred = self.discriminator(fake_images).view(-1)
        
        # Generator loss: BCE between fake predictions and real labels
        gen_loss = self.criterion(fake_pred, label_real)
        
        return gen_loss
    
class DiscriminatorLoss(torch.nn.Module):
    """
    Discriminator loss for GAN training.
    The discriminator tries to correctly classify real images as real 
    and fake images as fake.
    """
    def __init__(self, discriminator, real_image_key="inp", generated_image_key="generated_image"):
        super(DiscriminatorLoss, self).__init__()
        self.discriminator = discriminator
        self.real_image_key = real_image_key
        self.generated_image_key = generated_image_key
        self.criterion = nn.BCELoss()
        
    def forward(self, batch, model_outs):
        # Get real images from batch and fake images from generator output
        real_images = batch[self.real_image_key]
        fake_images = model_outs[self.generated_image_key]
        
        # Create labels
        label_real = torch.ones(real_images.size(0), device=real_images.device)
        label_fake = torch.zeros(fake_images.size(0), device=fake_images.device)
        
        # Get discriminator predictions
        # Detach fake images so gradients don't backprop into generator
        real_pred = self.discriminator(real_images).view(-1)
        fake_pred = self.discriminator(fake_images.detach()).view(-1)
        
        # Discriminator loss on real batch
        errD_real = self.criterion(real_pred, label_real)
        
        # Discriminator loss on fake batch
        errD_fake = self.criterion(fake_pred, label_fake)
        
        # Total discriminator loss
        disc_loss = errD_real + errD_fake
        
        return disc_loss
   


class GANLoss(nn.Module):
    """
    Combined GAN loss that wraps both generator and discriminator losses.
    This class creates a discriminator and provides both loss functions
    for adversarial training.
    """
    def __init__(self, real_image_key="inp", generated_image_key="generated_image"):
        super(GANLoss, self).__init__()
        self.discriminator = Discriminator()
        self.real_image_key = real_image_key
        self.generated_image_key = generated_image_key

        self.generator_loss = GeneratorLoss(
            self.discriminator, 
            generated_image_key=generated_image_key
        )
        self.discriminator_loss = DiscriminatorLoss(
            self.discriminator, 
            real_image_key=real_image_key, 
            generated_image_key=generated_image_key
        )



class LogTrainOutputs:
    def __init__(self):
        pass

    def run(self, model, dataset, out_dir, iter, model_outputs , batch ):
        # Log audio outputs for the first item in the batch

        real_images = batch['inp'] 
        generated_images = model_outputs['generated_image'] 

        fake_grid = vutils.make_grid(generated_images, padding=2, normalize=True)
        real_grid = vutils.make_grid(real_images, padding=2, normalize=True)

        side_by_side = torch.cat((real_grid, fake_grid), 2)  # Concatenate along width

        show_tensor(side_by_side)



        
