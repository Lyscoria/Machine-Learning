import os
import torch
import pickle
import numpy as np
from argparse import ArgumentParser

from Autoencoder import Autoencoder, AE_ENCODING_DIM
from VarAutoencoder import VarAutoencoder, VAE_ENCODING_DIM
from MLPAutoencoder import MLPAutoencoder, MLPAE_ENCODING_DIM
from utils.data_processor import create_flower_dataloaders, show_recover_results


parser = ArgumentParser()
parser.add_argument("--model", type=str, default="MLPAE", choices=["VAE", "AE", "MLPAE"])
args = parser.parse_args()

# set random seed
np.random.seed(0)
torch.manual_seed(0)

data_root = "./flowers"
vis_root = "./vis"
os.makedirs(vis_root, exist_ok=True)
model_ckpt = f"./model/Best_{args.model}.pth"
img_width, img_height = 24, 24
batch_size = 16

device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
print(f"using device {device}")


training_dataloader, validation_dataloader = create_flower_dataloaders(batch_size, data_root, img_width, img_height)

model = Autoencoder(AE_ENCODING_DIM) if args.model == 'AE' \
    else VarAutoencoder(VAE_ENCODING_DIM) if args.model == 'VAE' \
    else MLPAutoencoder(MLPAE_ENCODING_DIM, img_width, img_height)
model.load_state_dict(torch.load(model_ckpt))
model.eval()
model.to(device)

# visualize the results
train_images_sampled = next(iter(training_dataloader))[0].to(device)
valid_images_sampled = next(iter(validation_dataloader))[0].to(device)

train_outputs = model(train_images_sampled)
valid_outputs = model(valid_images_sampled)

if args.model == "VAE":
    train_outputs = train_outputs[0]
    valid_outputs = valid_outputs[0]

show_recover_results(train_images_sampled, train_outputs, f"{vis_root}/train_{args.model}.png")
show_recover_results(valid_images_sampled, valid_outputs, f"{vis_root}/valid_{args.model}.png")