import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from .nearest_embedding import NearestEmbed
from util.torchkit.networks import FlattenMlp
from torch.optim.lr_scheduler import CosineAnnealingLR

class VQ_VAE(nn.Module):
    def __init__(self,
            obs_dim,
            n_code,
            code_dim,
            lr,
            max_steps=100000,
            hidden_sizes=[256],
            device='cuda',
            share_codebook=False,
             vq_coef=0.0,
             commit_coef=0.4,
            init_emb = None,
            maxs=None,
            mins=None,
                 ):
        super(VQ_VAE, self).__init__()

        self.obs_dim = obs_dim
        self.share_codebook = share_codebook
        self.maxs=maxs
        self.mins=mins
        if share_codebook:
            self.n_code = n_code
        else:
            self.n_code = obs_dim * n_code
            self.n_code_each = n_code

        self.code_dim = code_dim
        self.rep_dim = self.obs_dim * self.code_dim
        self.hidden_sizes = hidden_sizes
        self.lr = lr
        self.obs_encoder = CustomLinear(
            input_size=obs_dim, code_dim=code_dim
        )
        self.emb = NearestEmbed(num_embeddings=self.n_code, embeddings_dim=self.code_dim, share_codebook=self.share_codebook, init_weight=init_emb)

        self.obs_decoder = FlattenMlp(
            input_size=code_dim * obs_dim, output_size=obs_dim, hidden_sizes=hidden_sizes
        )

        self.optimizer = torch.optim.Adam(self.parameters(), self.lr)
        self.lr_schedule = CosineAnnealingLR(self.optimizer, max_steps)
        self.to(device)

        self.vq_coef = vq_coef
        self.commit_coef = commit_coef
        self.mse = 0
        self.vq_loss = torch.zeros(1)
        self.commit_loss = 0



        #self.emb.weight.detach().normal_(0, 0.02)
        #torch.fmod(self.emb.weight, 0.04)

    def get_z_e(self, obs):
        if len(obs.shape)==1:
            batch_size = 1
        else:
            batch_size = obs.shape[0]
        obs = obs.reshape(batch_size, self.obs_dim)
        #one_hot_vectors = torch.eye(self.obs_dim, device=obs.device).unsqueeze(0).expand(batch_size, -1, -1)
        emb_hat = self.obs_encoder(obs)
        #return h2.view(-1, self.emb_size, int(self.hidden / self.emb_size))
        return emb_hat.permute(0,2,1)

    def decode(self, z_q):
        z_q = z_q.permute(0,2,1).reshape(-1, self.rep_dim)
        return self.obs_decoder(z_q)

    def forward(self, obs):
        z_e = self.get_z_e(obs)
        #z_q, _ = self.emb(z_e, weight_sg=True)
        z_q, _ = self.emb(z_e, weight_sg=True)
        emb, _ = self.emb(z_e.detach())
        return self.decode(z_q), z_e, emb

    def sample(self, size):
        sample = torch.randn(size, self.emb_size,
                             int(self.hidden / self.emb_size))
        if self.cuda():
            sample = sample.cuda()
        emb, _ = self.emb(sample)
        sample = self.decode(emb(sample).view(-1, self.hidden)).cpu()
        return sample

    def loss_function(self, x, recon_x, z_e, emb):
        self.ce_loss = F.binary_cross_entropy(recon_x, x.view(-1, 784))
        self.vq_loss = F.mse_loss(emb, z_e.detach())
        self.commit_loss = F.mse_loss(z_e, emb.detach())

        return self.ce_loss + self.vq_coef * self.vq_loss + self.comit_coef * self.commit_loss

    def latest_losses(self):
        return {'cross_entropy': self.ce_loss, 'vq': self.vq_loss, 'commitment': self.commit_loss}

    def update(self, obs, **kwargs):
        recon, z_e, emb = self.forward(obs)
        ce_loss = F.mse_loss(recon, obs)
        vq_loss = F.mse_loss(emb, z_e.detach())
        commit_loss = F.mse_loss(z_e, emb.detach())

        loss = ce_loss + self.vq_coef * vq_loss + self.commit_coef * commit_loss


        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.obs_encoder.safe_update()
        self.lr_schedule.step()
        results = {
            'loss': float(loss.detach().cpu().mean()),
            'recon_loss': float(ce_loss.detach().cpu().mean()),
            'code_loss': float(vq_loss.detach().cpu().mean()),
            'commit_loss': float(commit_loss.detach().cpu().mean())
        }

        return results

    def encode(self, obs, get_idx=False, clip=False, return_recon=False):
        if clip:
            # Expand self.mins and self.maxs to match the shape of obs
            #expanded_mins = self.mins.unsqueeze(0).expand_as(obs)
            #expanded_maxs = self.maxs.unsqueeze(0).expand_as(obs)

            # Clip the values
            obs = obs.clip(self.mins, self.maxs)
        z_e = self.get_z_e(obs)
        emb, argmin = self.emb(z_e.detach())
        emb = emb.permute(0,2,1).reshape(-1, self.rep_dim)
        if return_recon:
            #emb = self.obs_decoder(emb)  # test-1: reconstruction
            #emb = z_e.detach()   #test-2： continuous output
            #emb = self.tmp_encoder(obs)  #test-3: untrained encoder
            pass
        if not get_idx:
            return emb.detach()
        else:
            return emb.detach(), argmin

    def get_predicted_emb(self, obs, predicted_codes, mask):
        emb, labels = self.encode(obs.reshape(1, self.obs_dim), True)
        labels = labels.unsqueeze(0)
        labels[mask] = predicted_codes[mask]
        if not self.share_codebook:
            tmp = torch.arange(self.obs_dim, device=obs.device).unsqueeze(0).expand(1,-1) * self.n_code_each
            argmin = labels.clone()
            argmin += tmp

        result = self.emb.weight.t().index_select(0, argmin.view(-1)
                                      ).view(1,1,-1)

        return  result, labels

    def test_encoder_tmp(self):
        from util.util import mlp
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #dims = [self.obs_dim, self.obs_dim]
        #self.tmp_encoder =mlp( dims, squeeze_output=False)
        #self.tmp_encoder.to(device)
        #self.tmp_encoder = nn.Linear(self.obs_dim, self.obs_dim)
        self.tmp_encoder = CustomLinear(self.obs_dim)
        self.tmp_encoder.to(dtype=torch.float32)
        self.tmp_encoder.to(device)



class CustomLinear(nn.Module):
    def __init__(self, input_size, code_dim=1):
        super(CustomLinear, self).__init__()
        self.input_size = input_size
        self.code_dim = code_dim

        # Initialize weights and biases
        self.weights = nn.Parameter(torch.randn(input_size, code_dim) * 0.1 + 1)
        self.weights.data = torch.clamp(self.weights.data, min=0.8, max=1.2)

        self.bias = nn.Parameter(torch.randn(input_size, code_dim) * 0.1)
        self.bias.data = torch.clamp(self.bias.data, min=-0.2, max=0.2)

    def forward(self, x):
        # x has a shape of (batch_size, input_size, 1), we need to adjust it to perform element-wise multiplication
        # Expand x to match the (batch_size, input_size, code_dim) for dot multiplication

        x_expanded = x.unsqueeze(-1).expand(-1, -1, self.code_dim)  # Now x is (batch_size, input_size, code_dim)

        # Perform element-wise multiplication
        output = x_expanded * self.weights + self.bias  # Broadcasting will happen automatically

        return output

    def safe_update(self):
        self.weights.data = torch.clamp(self.weights.data, min=0.8, max=1.2)
        self.bias.data = torch.clamp(self.bias.data, min=-0.2, max=0.2)


