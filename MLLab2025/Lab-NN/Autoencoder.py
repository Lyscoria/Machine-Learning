import torch
import torch.nn as nn
import torch.nn.functional as F

AE_ENCODING_DIM = 64

# Define the Encoder
class Encoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Encoder, self).__init__()
        '''
        encoding_dim: the dimension of the latent vector produced by the encoder
        '''
        
        '''
        TODO: Implement the Encoder.

        Requirements:
        1. Use convolutional layers to extract features from the input images.
        2. Apply max pooling to downsample the spatial dimensions.
        3. Use a linear layer to map the feature maps to the latent vector.
        '''
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear = nn.Linear(128 * 6 * 6, encoding_dim)


    def forward(self, x):
        '''
        x: input images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        return v: latent vector, dim: (Batch_size, encoding_dim)
        '''
        
        '''
        TODO: Implement the forward pass of the Encoder.

        Steps:
        1. Pass the input through the convolutional layers and max pooling.
        2. Flatten the output and pass it through the linear layer to obtain the latent vector.
        3. Return the latent vector.
        '''
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = F.relu(x)
        
        x = x.view(x.size(0), -1)
        v = self.linear(x)
        return v


# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Decoder, self).__init__()
        '''
        encoding_dim: the dimension of the latent vector produced by the encoder
        '''
        
        '''
        TODO: Implement the Decoder.

        Requirements:
        1. Use a linear layer to map the latent vector back to the feature map dimensions.
        2. Use transposed convolutional layers to upsample the feature maps.
        3. Ensure the output has the same dimensions as the input image.
        '''
        self.linear = nn.Linear(encoding_dim, 128 * 6 * 6)
        self.deconv1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.deconv3 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=2, stride=2)


    def forward(self, v):
        '''
        v: latent vector, dim: (Batch_size, encoding_dim)
        return x: reconstructed images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        '''
        
        '''
        TODO: Implement the forward pass of the Decoder.

        Steps:
        1. Pass the latent vector through the linear layer to reconstruct the feature maps.
        2. Pass the feature maps through transposed convolutional layers to upsample them.
        3. Return the reconstructed images.
        '''
        x = F.relu(self.linear(v))
        x = x.view(x.size(0), 128, 6, 6)

        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        return x


# Combine the Encoder and Decoder to make the autoencoder
class Autoencoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(encoding_dim)
        self.decoder = Decoder(encoding_dim)

    def forward(self, x):
        '''
        TODO: Implement the forward pass of the Autoencoder.

        Steps:
        1. Pass the input through the Encoder to obtain the latent vector.
        2. Pass the latent vector through the Decoder to reconstruct the input.
        3. Return the reconstructed images.
        '''
        v = self.encoder(x)
        x = self.decoder(v)
        return x
    
    @property
    def name(self):
        return "AE"