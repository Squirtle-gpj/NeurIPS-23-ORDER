import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ObsRepMLP(nn.Module):
    def __init__(self, obs_dim, emb_dim=128, token_dim=3, both=False):
        super(ObsRepMLP, self).__init__()
        self.obs_dim = obs_dim
        self.token_dim = token_dim
        self.emb_dim = emb_dim
        self.attribute_dim = 1

        self.token_encoder = nn.Sequential(
            nn.Linear(self.attribute_dim, self.token_dim),
        )

        self.encoder_input_dim = obs_dim * token_dim
        self.encoder = nn.Sequential(
            nn.Linear(self.encoder_input_dim, self.emb_dim),
            nn.ReLU(),
        )
        self.both = both
        if both:
            self.encoder2 = nn.Sequential(
                nn.Linear(self.encoder_input_dim, self.emb_dim),
                nn.ReLU(),
            )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        torch.nn.init.normal_(self.mask_token, std=.02)

        #self.set_observable_type(observable_type=observable_type)

    def set_observable_type(self, observable_type='full'):
        self.observable_type = observable_type

    def forward(self, obs,  mask_scheme=None, device="cuda",):

        T, batch_size, _ = obs.shape
        new_batch_size = T * batch_size
        obs = obs.reshape(-1, 1).to(device)
        obs_token = self.token_encoder(obs)
        obs_token = obs_token.reshape(new_batch_size, self.obs_dim, self.token_dim)
        obs_token, mask = self.mask(obs_token, new_batch_size, mask_scheme)
        z = self.encoder(obs_token.reshape(new_batch_size, -1))


        if self.both:
            z2 = self.encoder2(obs_token.reshape(new_batch_size, -1))
            return z.reshape(T, batch_size, -1), z2.reshape(T, batch_size, -1), mask.reshape(T, batch_size, -1)

        return z.reshape(T, batch_size, -1), mask.reshape(T, batch_size, -1)

    def get_mask_entry(self):
        if self.observable_type == 'full':
            return None
        elif self.observable == 'random_episode':
            pass
        elif self.observable_type == 'random_step':
            pass
        elif self.observable_type == 'fixed':
            pass

    def mask(self, obs_token, batch_size, mask_scheme=None):
        if mask_scheme is None:
            mask_scheme = {
                'observable_type': 'full',
            }
        observable_type = mask_scheme.get('observable_type', 'full')
        mask_entry = mask_scheme.get('mask_entry', None)
        mask_ratio = mask_scheme.get('mask_ratio', None)

        # Initialize the mask tensor to False
        mask = torch.zeros(obs_token.shape[0], self.obs_dim, dtype=torch.bool)

        if observable_type == 'fixed':
            mask_token = self.mask_token.expand(batch_size, len(mask_entry), -1)
            obs_token[:, mask_entry, :] = mask_token
            mask[:, mask_entry] = True
        elif observable_type == 'random_step':

            num_masked_elements_per_row = min(int(self.obs_dim * mask_ratio) + 1, self.obs_dim)
            mask_indices = torch.randint(0, self.obs_dim, (obs_token.shape[0], num_masked_elements_per_row))
            mask.scatter_(1, mask_indices, True)
            mask_token = self.mask_token.squeeze(dim=1).expand(mask.sum(), -1)
            obs_token[mask, :] = mask_token
        elif observable_type == 'random_episode':
            #num_masked_elements = math.ceil(self.obs_dim * mask_ratio)
            #mask_entry = torch.randperm(self.obs_dim)[:num_masked_elements]
            mask_token = self.mask_token.expand(batch_size, len(mask_entry), -1)
            obs_token[:, mask_entry, :] = mask_token
            mask[:, mask_entry] = True

        return obs_token, mask


class ObsRepEncoder(nn.Module):
    def __init__(self, obs_dim, token_dim=32, forwarding_dim=128, n_head=4, n_layer=2, observable_type='full'):
        super(ObsRepEncoder, self).__init__()
        self.obs_dim = obs_dim
        self.token_dim = token_dim
        self.forwarding_dim = forwarding_dim
        self.attribute_dim = 1
        self.n_head = n_head
        self.n_layer = n_layer

        self.token_encoder = nn.Sequential(
                nn.Linear(self.attribute_dim, self.token_dim),
                nn.BatchNorm1d(self.token_dim),
            )



        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.token_dim, nhead=self.n_head, dim_feedforward=self.forwarding_dim,
                                                               dropout=0.1,)  # input_shape (seq, batch, feature)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.n_layer)



        self.rep_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.pos_emb = nn.Parameter(torch.zeros(1, obs_dim + 1, token_dim))  # fixed sin-cos embedding

        torch.nn.init.normal_(self.rep_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.pos_emb, std=.02)

        #self.observable_type = observable_type
        self.set_observable_type(observable_type=observable_type)

    def set_observable_type(self, observable_type='full'):
        self.observable_type = observable_type


    def forward(self, obs, device="cuda"):
        T, batch_size, _ = obs.shape
        new_batch_size = T*batch_size
        obs = obs.reshape(-1, 1).to(device)
        obs_token = self.token_encoder(obs)
        obs_token = obs_token.reshape(new_batch_size, self.obs_dim, self.token_dim)
        obs_token = self.mask(obs_token, new_batch_size)

        rep_token = self.rep_token.expand(new_batch_size, -1, -1)
        x = torch.cat([rep_token, obs_token], dim=1)
        pos_emb = self.pos_emb.expand(new_batch_size, -1, -1)
        x = pos_emb + x

        z = self.encoder(x.permute(1,0,2)) #seq,batch,feature

        #z = torch.mean(z,dim=0) #?
        z = z[0, :, :]

        return z.reshape(T, batch_size,-1)


    def get_mask_entry(self):
        if self.observable_type == 'full':
            return None
        elif self.observable == 'random_episode':
            pass
        elif self.observable_type == 'random_step':
            pass
        elif self.observable_type == 'fixed':
            pass

    def mask(self, obs_token, batch_size):
        mask_entry = self.get_mask_entry()
        if mask_entry is not None:
            mask_token = self.mask_token.expand(batch_size, len(mask_entry), -1)
            obs_token[:,mask_entry,:] = mask_token

        return obs_token

