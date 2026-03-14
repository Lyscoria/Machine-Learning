import os
import sys
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from argparse import ArgumentParser

from utils.train import train
from utils.scheduler import exponential_decay
from utils.data_processor import create_flower_dataloaders

from MLPAutoencoder import MLPAutoencoder, MLPAE_ENCODING_DIM
from VarAutoencoder import VarAutoencoder, VAE_loss_function, VAE_ENCODING_DIM
from Autoencoder import Autoencoder, AE_ENCODING_DIM

parser = ArgumentParser()
parser.add_argument("--model", type=str, default="AE", choices=["VAE", "AE", "MLPAE"])
args = parser.parse_args()

# Set seeds
torch.manual_seed(0)
np.random.seed(0)


data_root = "./flowers"
model_save_root = "./model"

batch_size = 16
num_epochs = 100
early_stopping_patience = 5
img_width, img_height = 24, 24
lr = 1. 

model = Autoencoder(AE_ENCODING_DIM) if args.model == 'AE' \
    else VarAutoencoder(VAE_ENCODING_DIM) if args.model == 'VAE' \
    else MLPAutoencoder(MLPAE_ENCODING_DIM, img_width, img_height)
loss_fn = VAE_loss_function if args.model == 'VAE' else F.mse_loss

optimizer = optim.SGD(model.parameters(), momentum=0., lr=lr) # TODO: You can change this in Part 3 Step 2, for faster and better convergence.
scheduler = exponential_decay(initial_learning_rate=lr, decay_rate=0.9, decay_epochs=10) # TODO: You can change this in Part 3 Step 2, for faster and better convergence.

if args.model == 'AE' or args.model == 'VAE':
    batch_size = 16
    num_epochs = 100
    early_stopping_patience = 5
    img_width, img_height = 24, 24
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lambda epoch: lr if epoch < 30 else lr * 0.1

training_dataloader, validation_dataloader = create_flower_dataloaders(batch_size, data_root, img_width, img_height)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
model.to(device)


# Start training
train(
    optimizer=optimizer,
    scheduler=scheduler,
    model=model,
    training_dataloader=training_dataloader,
    validation_dataloader=validation_dataloader,
    num_epochs=num_epochs,
    early_stopping_patience=early_stopping_patience,
    device=device,
    model_save_root=model_save_root,
    loss_fn=loss_fn,
)







