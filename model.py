"""Image-to-stroke model: CNN encoder + LSTM MDN decoder.

Architecture follows the stroke generation approach from Decoupled Style Descriptors
(Atarsaikhan et al.), adapted for image-conditioned generation:

- Encoder: ResNet-18 backbone → feature vector (replaces the style extraction network)
- Decoder: 2-layer LSTM with MDN head (same as DSD generation network)
  - Input at each step: previous (dx, dy, pen) + image feature
  - Output: K Gaussian mixture params for (dx, dy) + Bernoulli for pen state

MDN output per step (K=20 mixtures):
  - pi:   (K,)  mixture weights
  - mu:   (K,2) means
  - sigma:(K,2) standard deviations
  - rho:  (K,)  correlation
  - eos:  (1,)  pen-up logit
"""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


class ImageEncoder(nn.Module):
    """ResNet-18 encoder: image → feature vector."""

    def __init__(self, feature_dim: int = 256):
        super().__init__()
        resnet = models.resnet18(weights=None)
        # Replace first conv for potentially small images (64px height)
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()  # skip aggressive downsampling
        # Remove classification head
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # up to avgpool
        self.fc = nn.Linear(512, feature_dim)

    def forward(self, x):
        """x: (B, 3, H, W) → (B, feature_dim)"""
        feat = self.backbone(x).squeeze(-1).squeeze(-1)  # (B, 512)
        return self.fc(feat)  # (B, feature_dim)


class StrokeMDNDecoder(nn.Module):
    """LSTM decoder with MDN head for autoregressive stroke generation.

    Follows the DSD generation architecture:
    - LSTM1 processes stroke input
    - LSTM2 takes LSTM1 output concatenated with conditioning vector
    - FC layer outputs MDN parameters
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_mixtures: int = 20,
        num_layers: int = 2,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_mixtures = num_mixtures
        self.num_layers = num_layers

        # Stroke input projection (dx, dy, pen) → feature_dim
        self.input_fc = nn.Linear(3, feature_dim)
        self.input_relu = nn.LeakyReLU(0.1)

        # Two-stage LSTM (following DSD)
        self.lstm1 = nn.LSTM(feature_dim, feature_dim, num_layers=num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(feature_dim * 2, feature_dim * 2, num_layers=num_layers, batch_first=True)

        # MDN output: 1 (eos logit) + K*6 (mu1, mu2, sig1, sig2, rho, pi)
        self.mdn_fc = nn.Linear(feature_dim * 2, num_mixtures * 6 + 1)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, stroke_in, condition):
        """Teacher-forced forward pass.

        Args:
            stroke_in: (B, T, 3) input strokes (shifted target)
            condition:  (B, feature_dim) image feature vector

        Returns:
            mdn_params: dict with keys:
                eos_logit: (B, T, 1)
                mu:        (B, T, K, 2)
                sigma:     (B, T, K, 2)
                rho:       (B, T, K)
                pi:        (B, T, K)
        """
        B, T, _ = stroke_in.shape
        K = self.num_mixtures

        # Project stroke input
        x = self.input_fc(stroke_in)
        x = self.input_relu(x)

        # LSTM1
        out1, _ = self.lstm1(x)

        # Concatenate with conditioning (broadcast across time)
        cond_expanded = condition.unsqueeze(1).expand(B, T, self.feature_dim)
        lstm2_in = torch.cat([out1, cond_expanded], dim=-1)

        # LSTM2
        out2, _ = self.lstm2(lstm2_in)

        # MDN output
        mdn_out = self.mdn_fc(out2)  # (B, T, K*6+1)

        return self._parse_mdn_output(mdn_out)

    def _parse_mdn_output(self, mdn_out):
        """Parse raw MDN output into structured params."""
        K = self.num_mixtures
        eos_logit = mdn_out[..., 0:1]  # (B, T, 1)
        params = mdn_out[..., 1:]       # (B, T, K*6)

        mu1, mu2, sig1, sig2, rho, pi_logits = torch.split(params, K, dim=-1)

        sigma1 = torch.exp(sig1) + 1e-3
        sigma2 = torch.exp(sig2) + 1e-3
        rho = self.tanh(rho)
        pi = self.softmax(pi_logits)

        mu = torch.stack([mu1, mu2], dim=-1)       # (B, T, K, 2)
        sigma = torch.stack([sigma1, sigma2], dim=-1)  # (B, T, K, 2)

        return {
            "eos_logit": eos_logit,
            "mu": mu,
            "sigma": sigma,
            "rho": rho,
            "pi": pi,
        }

    def sample(self, condition, max_len: int = 300, temperature: float = 1.0):
        """Autoregressive sampling (no teacher forcing).

        Args:
            condition: (B, feature_dim) image features
            max_len: maximum sequence length
            temperature: scales sigma for diversity

        Returns:
            strokes: (B, T, 3) sampled trajectories
            lengths: (B,) actual lengths
        """
        B = condition.shape[0]
        device = condition.device

        # Init hidden states
        h1 = torch.zeros(self.num_layers, B, self.feature_dim, device=device)
        c1 = torch.zeros(self.num_layers, B, self.feature_dim, device=device)
        h2 = torch.zeros(self.num_layers, B, self.feature_dim * 2, device=device)
        c2 = torch.zeros(self.num_layers, B, self.feature_dim * 2, device=device)

        # Start token: zeros
        inp = torch.zeros(B, 1, 3, device=device)

        all_strokes = []
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for t in range(max_len):
            x = self.input_fc(inp)
            x = self.input_relu(x)

            out1, (h1, c1) = self.lstm1(x, (h1, c1))
            lstm2_in = torch.cat([out1, condition.unsqueeze(1)], dim=-1)
            out2, (h2, c2) = self.lstm2(lstm2_in, (h2, c2))

            mdn_out = self.mdn_fc(out2)
            params = self._parse_mdn_output(mdn_out)

            # Sample from MDN
            pi = params["pi"].squeeze(1)          # (B, K)
            mu = params["mu"].squeeze(1)          # (B, K, 2)
            sigma = params["sigma"].squeeze(1) * temperature
            rho = params["rho"].squeeze(1)        # (B, K)
            eos_logit = params["eos_logit"].squeeze(1)  # (B, 1)

            # Pick mixture component
            comp = torch.multinomial(pi, 1).squeeze(-1)  # (B,)

            # Sample (dx, dy) from chosen Gaussian
            batch_idx = torch.arange(B, device=device)
            sel_mu = mu[batch_idx, comp]      # (B, 2)
            sel_sigma = sigma[batch_idx, comp]  # (B, 2)
            sel_rho = rho[batch_idx, comp]    # (B,)

            # Correlated bivariate normal
            z1 = torch.randn(B, device=device)
            z2 = torch.randn(B, device=device)
            dx = sel_mu[:, 0] + sel_sigma[:, 0] * z1
            dy = sel_mu[:, 1] + sel_sigma[:, 1] * (sel_rho * z1 + torch.sqrt(1 - sel_rho**2 + 1e-6) * z2)

            # Pen state
            pen_prob = torch.sigmoid(eos_logit.squeeze(-1))
            pen = (pen_prob > 0.5).float()

            stroke = torch.stack([dx, dy, pen], dim=-1)  # (B, 3)
            all_strokes.append(stroke)

            # Update finished
            finished = finished | (pen > 0.5).bool()

            # Next input
            inp = stroke.unsqueeze(1)

            if finished.all():
                break

        strokes = torch.stack(all_strokes, dim=1)  # (B, T, 3)
        lengths = torch.full((B,), strokes.shape[1], dtype=torch.long, device=device)
        return strokes, lengths


class InkStrokePredictor(nn.Module):
    """Full model: image → stroke trajectory."""

    def __init__(
        self,
        feature_dim: int = 256,
        num_mixtures: int = 20,
        num_layers: int = 2,
    ):
        super().__init__()
        self.encoder = ImageEncoder(feature_dim=feature_dim)
        self.decoder = StrokeMDNDecoder(
            feature_dim=feature_dim,
            num_mixtures=num_mixtures,
            num_layers=num_layers,
        )

    def forward(self, image, stroke_in):
        """Teacher-forced forward pass.

        Args:
            image:     (B, 3, H, W)
            stroke_in: (B, T, 3) input strokes (target shifted right by 1)

        Returns:
            mdn_params dict
        """
        features = self.encoder(image)
        return self.decoder(stroke_in, features)

    def sample(self, image, max_len=300, temperature=1.0):
        """Generate strokes from image."""
        features = self.encoder(image)
        return self.decoder.sample(features, max_len=max_len, temperature=temperature)
