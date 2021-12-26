"""
This is a slight modification of Shubham Chandel's implementation of a
variational autoencoder in PyTorch.
Source: https://github.com/sksq96/pytorch-vae
"""
import typing

import cv2 as cv
import tqdm
import numpy as np
import torch, torch.nn.functional
import wandb


class VAE(torch.nn.Module):
    """Input should be (bsz, C, H, W) where C=3, H=42, W=144"""

    def __init__(self, im_c=3, im_h=42, im_w=144, z_dim=32):
        super().__init__()

        self.im_c = im_c
        self.im_h = im_h
        self.im_w = im_w
        # (144, 42) -> (21, 72) -> (10, 36) -> (5, 18) -> (2, 9)
        encoder_list = [
            torch.nn.Conv2d(im_c, 32, kernel_size=(4, 4), stride=(2,), padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2,), padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2,), padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2,), padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        ]
        self.encoder = torch.nn.Sequential(*encoder_list)
        sample_img = torch.zeros([1, im_c, im_h, im_w])
        em_shape = torch.nn.Sequential(*encoder_list[:-1])(sample_img).shape[1:]
        h_dim: int = typing.cast(int, np.prod(em_shape))

        self.fc1 = torch.nn.Linear(h_dim, z_dim)
        self.fc2 = torch.nn.Linear(h_dim, z_dim)
        self.fc3 = torch.nn.Linear(z_dim, h_dim)

        self.decoder = torch.nn.Sequential(
            torch.nn.Unflatten(1, em_shape),
            torch.nn.ConvTranspose2d(
                em_shape[0],
                128,
                kernel_size=(4, 4),
                stride=(2,),
                padding=(1, 1),
                output_padding=(1, 0),
            ),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(
                128, 64, kernel_size=(4, 4), stride=(2,), padding=(1, 1)
            ),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(
                64,
                32,
                kernel_size=(4, 4),
                stride=(2,),
                padding=(1, 1),
                output_padding=(1, 0),
            ),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(
                32, im_c, kernel_size=(4, 4), stride=(2,), padding=(1, 1)
            ),
            torch.nn.Sigmoid(),
        )

    @staticmethod
    def reparameterize(mu, log_variable):
        std = log_variable.mul(0.5).exp_()
        esp = torch.randn(*mu.size(), device=mu.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, log_variable = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, log_variable)
        return z, mu, log_variable

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def encode_raw(self, x: np.ndarray, device):
        # assume x is RGB image with shape (bsz, H, W, 3)
        p = np.zeros([x.shape[0], 42, 144, 3], np.float)
        for i in range(x.shape[0]):
            p[i] = cv.resize(x[i], (144, 144))[68:110] / 255
        x = p.transpose((0, 3, 1, 2))
        x = torch.as_tensor(x, device=device, dtype=torch.float)
        v = self.representation(x)
        return v, v.detach().cpu().numpy()

    def encode(self, x):
        h = self.encoder(x)
        z, mu, log_variable = self.bottleneck(h)
        return z, mu, log_variable

    def decode(self, z):
        z = self.fc3(z)
        return self.decoder(z)

    def forward(self, x):
        # expects (N, C, H, W)
        z, mu, log_variable = self.encode(x)
        z = self.decode(z)
        return z, mu, log_variable

    @staticmethod
    def loss(actual, reconstructed, mu, log_variable, kld_weight=1.0):
        bce = torch.nn.functional.binary_cross_entropy(
            reconstructed, actual, reduction="sum"
        )
        kld = -0.5 * torch.sum(1 + log_variable - mu ** 2 - log_variable.exp())
        return bce + kld * kld_weight


if __name__ == "__main__":
    wandb.init(project="aicrowd-racing", name="vae-training-1", save_code=False)

    images = np.load("data/expert/images.npy")
    images = np.stack([cv.resize(image, (144, 144))[68:110] / 255 for image in images])
    images = images.transpose((0, 3, 1, 2))

    indices = np.random.permutation(images.shape[0])
    threshold = int(images.shape[0] * 0.9)
    train_indices, test_indices = indices[:threshold], indices[threshold:]
    visualize_indices = np.concatenate(
        [np.random.choice(train_indices, 5), np.random.choice(train_indices, 15)]
    )

    batch_size = 32
    learning_rate = 1e-3
    num_epochs = 25

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE().to(DEVICE)
    optim = torch.optim.Adam(vae.parameters(), lr=learning_rate)
    best_loss = 1e10

    def evaluate(epoch):
        global best_loss, test_indices
        test_indices = np.random.permutation(test_indices)
        test_loss = []
        vae.eval()
        with tqdm.trange(
            len(test_indices) // batch_size, desc=f"Epoch #{epoch} test"
        ) as iterator:
            for i in iterator:
                index = test_indices[batch_size * i : batch_size * (i + 1)]
                img = torch.as_tensor(images[index], device=DEVICE, dtype=torch.float)
                loss = vae.loss(img, *vae(img), kld_weight=0.0)
                test_loss.append(loss.item())
            test_loss = np.mean(test_loss)
        wandb.log({"vae/test_loss": test_loss, "epoch": epoch})

        if test_loss < best_loss:
            best_loss = test_loss
            print(f"Saving model at epoch #{epoch}")
            torch.save(vae.state_dict(), "data/models/vae_rgb_front.pth")

        for idx in visualize_indices:
            original_image = torch.as_tensor(
                images[idx], device=DEVICE, dtype=torch.float
            )
            reconstructed_image = vae(original_image[None])[0][0].detach().cpu().numpy()
            wandb.log(
                {
                    "vae/original_image": wandb.Image(
                        images[idx].transpose(1, 2, 0), caption=f"{idx}-original"
                    ),
                    "vae/reconstructed_image": wandb.Image(
                        reconstructed_image.transpose(1, 2, 0), caption=f"{idx}-vae"
                    ),
                    "epoch": epoch,
                }
            )

    def train(epoch):
        global train_indices
        train_indices = np.random.permutation(train_indices)
        train_loss = []
        vae.train()
        with tqdm.trange(
            len(train_indices) // batch_size, desc=f"Epoch #{epoch} train"
        ) as iterator:
            for i in iterator:
                index = train_indices[batch_size * i : batch_size * (i + 1)]
                img = torch.as_tensor(images[index], device=DEVICE, dtype=torch.float)
                loss = vae.loss(img, *vae(img))
                optim.zero_grad()
                loss.backward()
                optim.step()
                train_loss.append(loss.item())
            train_loss = np.mean(train_loss)
            wandb.log({"vae/train_loss": train_loss, "epoch": epoch})

    evaluate(0)
    for epoch_idx in range(1, num_epochs + 1):
        train(epoch_idx)
        evaluate(epoch_idx)
