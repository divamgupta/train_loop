
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

class MNISTDataset(torch.utils.data.Dataset):
    """
    MNIST dataset that returns dicts with keys 'inp' and 'gt'.
    """
    def __init__(self, train=True):
        transform = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.dataset = datasets.MNIST(
            root='/tmp/dataMN',
            train=train,
            download=True,
            transform=transform
        )
        self.collate_fn = None  # Use default collate function

    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        inp, gt = self.dataset[idx]
        return {"inp": inp, "gt_class_id": gt}


class Generator(nn.Module):
    def __init__(self, d=128):
        super(Generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 1, 4, 2, 1)

    def forward(self, batch ):
        device = batch['inp'].device
        image_batch = batch['inp']
        input = torch.randn(image_batch.size(0), 100, 1, 1, device=device)

        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = torch.tanh(self.deconv5(x))

        return {"generated_image": x }

class Discriminator(nn.Module):
    def __init__(self, d=128):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)

    def forward(self, input ):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = torch.sigmoid(self.conv5(x))

        return x


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
        
    def forward(self, batch, model_outs):
        # Get the generated (fake) images from generator output
        fake_images = model_outs[self.generated_image_key]
        
        # Create real labels - generator wants discriminator to classify fakes as real
        label_real = torch.ones(fake_images.size(0), device=fake_images.device)
        
        # Get discriminator's prediction on fake images
        fake_pred = self.discriminator(fake_images).squeeze()
        
        # Generator loss: BCE between fake predictions and real labels
        gen_loss = F.binary_cross_entropy(fake_pred, label_real)
        
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
        
    def forward(self, batch, model_outs):
        # Get real images from batch and fake images from generator output
        real_images = batch[self.real_image_key]
        fake_images = model_outs[self.generated_image_key]
        
        # Create labels
        label_real = torch.ones(real_images.size(0), device=real_images.device)
        label_fake = torch.zeros(fake_images.size(0), device=fake_images.device)
        
        # Get discriminator predictions
        # Detach fake images so gradients don't backprop into generator
        real_pred = self.discriminator(real_images).squeeze()
        fake_pred = self.discriminator(fake_images.detach()).squeeze()
        
        # Discriminator loss: average of BCE for real and fake
        disc_loss = 0.5 * (
            F.binary_cross_entropy(real_pred, label_real) +
            F.binary_cross_entropy(fake_pred, label_fake)
        )
        
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

    def _tensor_to_ascii(self, img_tensor):
        """Convert a tensor image to ASCII art for terminal display."""
        # Denormalize from [-1, 1] to [0, 1]
        img = (img_tensor.squeeze().detach().cpu() + 1) / 2
        # Resize to a smaller size for terminal display
        img = torch.nn.functional.interpolate(img.unsqueeze(0).unsqueeze(0), size=(16, 16), mode='nearest').squeeze()
        # ASCII characters from dark to light
        chars = " .:-=+*#%@"
        result = []
        for row in img:
            line = ""
            for val in row:
                idx = int(val.item() * (len(chars) - 1))
                idx = max(0, min(len(chars) - 1, idx))
                line += chars[idx] * 2  # Double width for aspect ratio
            result.append(line)
        return "\n".join(result)

    def run(self, model, dataset, out_dir, iter, model_outputs , batch ):
        # Log audio outputs for the first item in the batch

        real_images = batch['inp'][0:1]
        generated_images = model_outputs['generated_image'][0:1]

        print(f"\n=== Iteration {iter} ===")
        print("Real image:")
        print(self._tensor_to_ascii(real_images[0]))
        print("\nGenerated image:")
        print(self._tensor_to_ascii(generated_images[0]))
