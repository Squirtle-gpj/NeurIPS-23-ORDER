import torch
import torch.nn as nn
from util.util import mlp





import torch
import torch.nn as nn
from torch.nn import functional as F
from util import helpers as utl
from util.torchkit.constant import *
from util.torchkit.networks import FlattenMlp
from src.codebook import Codebook
from torch.optim.lr_scheduler import CosineAnnealingLR


class VQ_VAE_multi_factor(nn.Module):
    def __init__(
            self,
            obs_dim,
            n_code,
            code_dim,
            lr,
            max_steps=100000,
            hidden_sizes=[256, 256],
            device='cuda',
            share_codebook=False,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_code = n_code
        self.code_dim = code_dim
        self.hidden_sizes = hidden_sizes
        self.lr = lr
        self.obs_encoder = FlattenMlp(
            input_size=1+obs_dim, output_size=code_dim, hidden_sizes=hidden_sizes
        )

        self.share_codebook = share_codebook
        if share_codebook:
            self.codebook = Codebook(
                num_codebook=n_code, dim_codebook=code_dim
            )

        self.obs_decoder = FlattenMlp(
            input_size=code_dim*obs_dim, output_size=obs_dim, hidden_sizes=hidden_sizes
        )

        self.optimizer = torch.optim.Adam(self.parameters(), self.lr)
        self.policy_lr_schedule = CosineAnnealingLR(self.optimizer, max_steps)
        self.to(device)

    def set_encode_type(self, type='org'):
        self.encode_type = type

    def encode(self, obs, ):
        if self.encode_type == 'org':
            batch_size, _ = obs.shape
            obs = obs.reshape(batch_size, self.obs_dim, 1)
            one_hot_vectors = torch.eye(self.obs_dim, device=obs.device).unsqueeze(0).expand(batch_size, -1, -1)
            emb_hat = self.obs_encoder(torch.cat([obs, one_hot_vectors], dim=-1))
            quantized, _, _ = self.codebook.quantize(emb_hat.reshape(batch_size*self.obs_dim, -1))
            quantized = quantized.reshape(batch_size, -1)
            return quantized
        elif self.encode_type == 'identity':
            return obs
        elif self.encode_type == 'recon':
            return self.recon(obs)

    def decode(self, emb):
        return self.obs_decoder(emb)

    def recon(self, obs):
        quantized = self.encode(obs)
        reconstructions = self.decode(quantized)
        # reconstructions = torch.sign(reconstructions)*(torch.exp(reconstructions.abs())-1)
        # reconstructions[5:] = observations[5:]

        return reconstructions

    def quantitize(self, obs):
        batch_size, _ = obs.shape
        obs = obs.reshape(batch_size, self.obs_dim, 1)
        one_hot_vectors = torch.eye(self.obs_dim, device=obs.device).unsqueeze(0).expand(batch_size, -1, -1)
        emb_hat = self.obs_encoder(torch.cat([obs, one_hot_vectors], dim=-1))
        quantized, indicies_onehot, codebook_metrics = self.codebook.quantize(emb_hat.reshape(batch_size * self.obs_dim, -1))
        quantized = quantized.reshape(batch_size, -1)
        return quantized, codebook_metrics

    def update(self, obs, **kwargs):
        quantized, codebook_metrics = self.quantitize(obs)
        reconstructions = self.decode(quantized)
        #sym_log_obs = torch.sign(observations) * torch.log(observations.abs() + 1)  # symlog reconstruction used in dreamer v3
        #recon_loss = F.mse_loss(reconstructions, sym_log_obs)
        recon_loss = F.mse_loss(reconstructions, obs)
        code_loss = codebook_metrics["loss_latent"]

        loss = recon_loss + code_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        results = {
            'loss': float(loss.detach().cpu().mean()),
            'recon_loss': float(recon_loss.detach().cpu().mean()),
            'code_loss': float(code_loss.detach().cpu().mean()),
            'perplexity': float(codebook_metrics['perplexity'].detach().cpu().mean())
        }

        return results

class VQ_VAE(nn.Module):
    def __init__(
        self,
        obs_dim,
        n_code,
        code_dim,
        lr,
        max_steps=100000,
        hidden_sizes = [256, 256],
        device='cuda',
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_code = n_code
        self.code_dim = code_dim
        self.hidden_sizes = hidden_sizes
        self.lr = lr
        self.obs_encoder =  FlattenMlp(
            input_size=obs_dim, output_size=code_dim, hidden_sizes=hidden_sizes
        )
        self.codebook = Codebook(
            num_codebook=n_code, dim_codebook=code_dim
        )
        self.obs_decoder = FlattenMlp(
            input_size=code_dim, output_size=obs_dim, hidden_sizes=hidden_sizes
        )

        self.optimizer = torch.optim.Adam(self.parameters(), self.lr)
        self.policy_lr_schedule = CosineAnnealingLR(self.optimizer, max_steps)
        self.to(device)

    def set_encode_type(self, type='org'):
        self.encode_type = type

    def encode(self, obs,):
        if self.encode_type == 'org':
            emb_hat = self.obs_encoder(obs)
            quantized, _, _ = self.codebook.quantize(emb_hat)
            return quantized
        elif self.encode_type == 'identity':
            return obs
        elif self.encode_type == 'recon':
            return self.recon(obs)

    def decode(self, emb):
        return self.obs_decoder(emb)

    def recon(self, observations):
        emb_hat = self.obs_encoder(observations)
        quantized, _, codebook_metrics = self.codebook.quantize(emb_hat)
        reconstructions = self.decode(quantized)
        #reconstructions = torch.sign(reconstructions)*(torch.exp(reconstructions.abs())-1)
        #reconstructions[5:] = observations[5:]

        return reconstructions

    def quantitize(self, obs):
        emb_hat = self.obs_encoder(obs)
        quantized, codebook_metrics = self.codebook.quantize(emb_hat)
        return quantized, codebook_metrics

    def update(self, observations, **kwargs):
        emb_hat = self.obs_encoder(observations)
        quantized, _, codebook_metrics = self.codebook.quantize(emb_hat)
        reconstructions = self.decode(quantized)
        sym_log_obs = torch.sign(observations) * torch.log(observations.abs() + 1) #symlog reconstruction used in dreamer v3
        recon_loss = F.mse_loss(reconstructions, sym_log_obs)
        code_loss = codebook_metrics["loss_latent"]

        loss = recon_loss + code_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        results = {
            'loss': float(loss.detach().cpu().mean()),
            'recon_loss': float(recon_loss.detach().cpu().mean()),
            'code_loss': float(code_loss.detach().cpu().mean()),
            'perplexity': float(codebook_metrics['perplexity'].detach().cpu().mean())
        }

        return results

