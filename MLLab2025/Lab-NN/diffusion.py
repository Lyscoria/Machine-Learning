import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler

from utils.scheduler import exponential_decay
from utils.data_processor import create_flower_dataloaders


# Set random seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Basic settings
data_root = "./flowers"  # Path to the dataset root directory
model_save_path = "./model/Bestmodel_diffusion.pkl"  # Path to save the trained model
vis_root = "./vis"

# Hyperparameters (adjustable)
batch_size = 16  # Batch size for training and validation
num_epochs = 500  # Number of training epochs
img_channel = 3
img_width, img_height = 24, 24
num_classes = 3

device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
print(f"using device {device}")

# Diffusion process settings (adjustable)
num_steps = 500  # Number of steps in the diffusion process
beta_min = 1e-4  # Minimum value of the beta schedule
beta_max = 5e-2  # Maximum value of the beta schedule

# Initialize diffusion parameters
# These parameters define the noise schedule and are crucial for the forward/reverse process
betas = torch.linspace(beta_min, beta_max, num_steps, device=device)

alphas = 1 - betas  # Compute alpha values from beta
alphas_sqrt = torch.sqrt(alphas)  # Square root of alphas
alphas_bar = torch.cumprod(alphas, 0)  # Cumulative product of alphas over steps
alphas_bar_sqrt = torch.sqrt(alphas_bar)  # Square root of cumulative alphas
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_bar)  # Square root of 1 - cumulative alphas


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(max_period)) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        x_norm1 = (1 + scale_msa.unsqueeze(1)) * self.norm1(x) + shift_msa.unsqueeze(1)
        attn_output, _ = self.attn(x_norm1, x_norm1, x_norm1)
        x = x + gate_msa.unsqueeze(1) * attn_output
        
        x_norm2 = (1 + scale_mlp.unsqueeze(1)) * self.norm2(x) + shift_mlp.unsqueeze(1)
        mlp_output = self.mlp(x_norm2)
        x = x + gate_mlp.unsqueeze(1) * mlp_output
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = (1 + scale.unsqueeze(1)) * self.norm_final(x) + shift.unsqueeze(1)
        x = self.linear(x)
        return x


# Define the diffusion model
class Diffuser(nn.Module):
    def __init__(self, n_steps, img_channels=3, num_classes=3, base_channels=96):
        """
        Initialize a diffusion model. The model predicts the added noise at each time step.

        Args:
          - n_steps (int): Number of diffusion steps (t).
          - img_channels (int): Number of image channels (e.g., 3 for RGB images).
        """
        super(Diffuser, self).__init__()        
        self.patch_size = 2
        self.hidden_size = base_channels * 4
        self.depth = 6
        self.num_heads = 6
        
        self.n_steps = n_steps
        self.num_classes = num_classes

        self.x_embed = nn.Conv2d(img_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)
        
        num_patches = (img_width // self.patch_size) * (img_height // self.patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.hidden_size))

        self.t_embed = TimestepEmbedder(self.hidden_size)
        self.y_embed = nn.Embedding(num_classes, self.hidden_size)
        
        self.blocks = nn.ModuleList([
            DiTBlock(self.hidden_size, self.num_heads) for _ in range(self.depth)
        ])
        
        self.final_layer = FinalLayer(self.hidden_size, self.patch_size, img_channels)
        
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        w = self.x_embed.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embed.bias, 0)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        c = 3
        p = self.patch_size
        h = img_height // p
        w = img_width // p
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, labels):
        """
        Forward pass of the diffusion model:
        Predict the noise added to x at a given time step t.

        Input:
          - x (torch.Tensor): Noisy input image, shape (batch_size, img_channels, height, width).
          - t (torch.Tensor): Time step indices, shape (batch_size,).
        Output:
          - Predicted noise, shape same as x.
        """
        x = self.x_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        
        t_emb = self.t_embed(t)
        y_emb = self.y_embed(labels)
        c = t_emb + y_emb
        
        for block in self.blocks:
            x = block(x, c)
            
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        
        return x

    def sample(self, shape, n_steps, labels):
        """
        Generate images by iteratively sampling from pure noise.

        Args:
        - shape (tuple): Shape of the generated images (batch_size, img_channels, height, width).
        - n_steps (int): Number of diffusion steps.

        Returns:
        - x_seq (list): List of images at each step of the reverse process.
        """
        self.eval()

        x_t = torch.randn(shape, device=next(self.parameters()).device)
        x_seq = [x_t]
        
        dt = 1.0 / n_steps
        
        for i in range(n_steps):
            t = 1.0 - i * dt
            x_t = self.p_theta_sampling(x_t, t, labels, dt)
            x_seq.append(x_t)
        return x_seq

    def p_theta_sampling(self, x, t, labels, dt):
        """
        Estimate x[t-1] given x[t] using the reverse diffusion process.

        Steps:
        1. Predict the noise added to x[t] using the model.
        2. Compute the mean of the reverse process based on the formula.
        3. Optionally add Gaussian noise for stochasticity if t > 0.

        Args:
        - model (Diffuser): The diffusion model.
        - x (torch.Tensor): Image at time step t, shape (batch_size, img_channels, height, width).
        - t (int): Current time step.

        Returns:
        - x_t_minus_1 (torch.Tensor): Estimated image at time step t-1.
        """
        batch_size = x.size(0)
        t_tensor = torch.tensor([t] * batch_size, device=x.device) * 999
        v_pred = self.forward(x, t_tensor, labels)
        
        x_next = x - v_pred * dt
            
        return x_next
    
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

# Loss function for training
def diffusion_loss_fn(model, x_0, labels):
    """
    Compute the training loss for the diffusion model:
    This function compares the model's predicted noise with the actual noise.

    Steps:
      1. Randomly sample a batch of time steps t.
      2. Add Gaussian noise to x_0 to simulate the forward process.
      3. Predict the noise using the model.
      4. Compute the mean squared error between predicted and actual noise.

    Args:
      - model (Diffuser): The diffusion model.
      - x_0 (torch.Tensor): Original clean image, shape (batch_size, img_channels, height, width).

    Returns:
      - loss (torch.Tensor): Scalar value representing the loss.
    """
    batch_size = x_0.size(0)
    
    t = torch.rand(batch_size, device=device)
    x_1 = torch.randn_like(x_0)
    
    t_broad = t.view(-1, 1, 1, 1)
    x_t = t_broad * x_1 + (1 - t_broad) * x_0
    
    target_v = x_1 - x_0
    
    pred_v = model(x_t, t * 999, labels)
    loss = F.mse_loss(pred_v, target_v)
    
    return loss

# Sampling function with visualization
def visualize_sampling(model, sample_shape, n_steps, save_path, target_labels=None):
    """Visualize the sampling process during training."""
    model.eval()

    current_batch_size = sample_shape[0]
    
    if target_labels is not None:
        if isinstance(target_labels, int):
            labels = torch.tensor([target_labels] * current_batch_size, device=device).long()
        else:
            labels = torch.tensor(target_labels, device=device).long()
    else:
        labels = torch.randint(low=0, high=num_classes, size=(current_batch_size,), device=device).long()

    with torch.no_grad():
        x_seq = model.sample(sample_shape, n_steps, labels)
    
    num_shows = 10  # Number of steps to display
    step_interval = n_steps // num_shows

    rows = current_batch_size
    cols = num_shows

    fig, axes = plt.subplots(rows, cols, figsize=(15, 2 * rows))

    if rows == 1: 
        axes = axes[None, :]
    
    for row in range(rows):
        for i in range(cols):
            ax = axes[row, i]
            step = i * step_interval
            
            if step >= len(x_seq): step = len(x_seq) - 1

            image = x_seq[step][row].squeeze().permute(1, 2, 0).detach().cpu().numpy()
            
            image_min, image_max = image.min(), image.max()
            normalized_image = 2 * (image - image_min) / (image.max() - image.min()) - 1
            normalized_image = normalized_image.clip(-1, 1)
            imshow_image = (normalized_image + 1) / 2
            
            ax.imshow(imshow_image)
            ax.axis('off')
            
            if row == 0: 
                ax.set_title(f"Step {step}")
            
            if i == 0:
                ax.text(-5, image.shape[0]//2, f"Class {labels[row].item()}", 
                        rotation=90, verticalalignment='center', fontweight='bold')

    plt.suptitle(f"Sampling Visualization")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    # Load data
    training_dataloader, validation_dataloader = create_flower_dataloaders(
        batch_size, data_root, img_width, img_height
    )

    # Initialize model, optimizer, and scheduler
    model = Diffuser(n_steps=num_steps, num_classes=num_classes, base_channels=96).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = exponential_decay(initial_learning_rate=1e-4, decay_rate=0.9, decay_epochs=5)
    scaler = GradScaler('cuda')

    sample_shape = (3, img_channel, img_width, img_height)  # Shape for visualization samples

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for images, labels in training_dataloader:
            images = images.to(device).float()
            labels = labels.to(device).long()
            # normalize images input to [-1, 1]
            images = 2 * images - 1
            
            optimizer.zero_grad()

            with autocast('cuda'):
                loss = diffusion_loss_fn(model, images, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())
        
        avg_train_loss = sum(train_losses) / len(train_losses)

        print(f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

        # Visualize sampling process every 10 epochs
        if (epoch + 1) % 10 == 0:
            visualize_sampling(model, sample_shape, num_steps, os.path.join(vis_root, f"random_images_diffusion_epoch_{epoch + 1}"), target_labels=[0, 1, 2])

    model.save(model_save_path)
    visualize_sampling(model, sample_shape, num_steps, os.path.join(vis_root, "random_images_diffusion"), target_labels=[0, 1, 2])