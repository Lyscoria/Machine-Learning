import torch
from torch import nn
from torch.nn import functional as F

VAE_ENCODING_DIM = 64

# Define the Variational Encoder
class VarEncoder(nn.Module):
    def __init__(self, encoding_dim):
        '''
        encoding_dim: the dimension of the latent vector produced by the encoder
        '''
        super(VarEncoder, self).__init__()
        # TODO: implement the encoder

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        flatten_dim = 128 * 6 * 6
        self.linear_mu = nn.Linear(flatten_dim, encoding_dim)
        self.linear_log_var = nn.Linear(flatten_dim, encoding_dim)

    def forward(self, x):
        '''
        x: input images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        return mu: mean of the distribution, dim: (Batch_size, encoding_dim)
        return log_var: log of the variance of the distribution, dim: (Batch_size, encoding_dim)
        '''
        
        # TODO: implement the forward pass

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        mu = self.linear_mu(x)
        log_var = self.linear_log_var(x)
        
        return mu, log_var

# Define the Decoder
class VarDecoder(nn.Module):
    def __init__(self, encoding_dim):
        '''
        encoding_dim: the dimension of the latent vector produced by the encoder
        '''
        super(VarDecoder, self).__init__()
        # TODO: implement the decoder

        flatten_dim = 128 * 6 * 6
        self.linear = nn.Linear(encoding_dim, flatten_dim)
        self.deconv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2)

    def forward(self, v):
        '''
        v: latent vector, dim: (Batch_size, encoding_dim)
        return x: reconstructed images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        '''
        
        # TODO: implement the forward pass

        x = F.relu(self.linear(v))
        x = x.view(x.size(0), 128, 6, 6)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        return x

# Define the Variational Autoencoder
class VarAutoencoder(nn.Module):
    def __init__(self, encoding_dim):
        super(VarAutoencoder, self).__init__()
        self.encoder = VarEncoder(encoding_dim)
        self.decoder = VarDecoder(encoding_dim)

    @property
    def name(self):
        return "VAE"

    def reparameterize(self, mu, log_var):
        '''
        mu: mean of the distribution, dim: (Batch_size, encoding_dim)
        log_var: log of the variance of the distribution, dim: (Batch_size, encoding_dim)
        return v: sampled latent vector, dim: (Batch_size, encoding_dim)
        '''
        # TODO: implement the reparameterization trick to sample v

        if self.training:
            sigma = torch.exp(0.5 * log_var)
            eps = torch.randn_like(sigma)
            z = mu + sigma * eps
        else:
            z = mu

        return z
        
    def forward(self, x):
        '''
        x: input images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        return x: reconstructed images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        return mu: mean of the distribution, dim: (Batch_size, encoding_dim)
        return log_var: log of the variance of the distribution, dim: (Batch_size, encoding_dim)
        '''
        # TODO: implement the forward pass
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x = self.decoder(z)
        return x, mu, log_var

# Loss Function
def VAE_loss_function(outputs, images):
    '''
    outputs: (x, mu, log_var)
    images: input/original images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
    return loss: the loss value, dim: (1)
    '''
    # TODO: implement the loss function for VAE

    recon_loss = F.mse_loss(outputs[0], images, reduction='sum')
    KL_divergence = -0.5 * torch.sum(1 + outputs[2] - outputs[1].pow(2) - outputs[2].exp())
    return recon_loss + KL_divergence

