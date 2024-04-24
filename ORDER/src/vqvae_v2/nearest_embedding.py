import numpy as np
import torch
from torch import nn
from torch.autograd import Function, Variable
import torch.nn.functional as F


class NearestEmbedFunc(Function):
    """
    Input:
    ------
    x - (batch_size, emb_dim, *)
        Last dimensions may be arbitrary
    emb - (emb_dim, num_emb)
    """
    @staticmethod
    def forward(ctx, input, emb, share_codebook=True):
        if input.size(1) != emb.size(0):
            raise RuntimeError('invalid argument: input.size(1) ({}) must be equal to emb.size(0) ({})'.
                               format(input.size(1), emb.size(0)))

        # save sizes for backward
        ctx.batch_size = input.size(0)
        ctx.num_latents = int(np.prod(np.array(input.size()[2:])))
        ctx.emb_dim = emb.size(0)
        ctx.num_emb = emb.size(1)
        ctx.input_type = type(input)
        ctx.dims = list(range(len(input.size())))

        B, D, O = input.shape

        # expand to be broadcast-able
        x_expanded = input.unsqueeze(-1) #B, D, O, 1
        if share_codebook:
            _, K = emb.shape
            emb_expanded = emb.unsqueeze(dim=1)  #D,1,K
        else:
            emb_expanded = emb.reshape(D,O,-1) #D,O,K
            _,_,K = emb_expanded.shape
        """
        num_arbitrary_dims = len(ctx.dims) - 2
        if num_arbitrary_dims:
            emb_expanded = emb.view(
                emb.shape[0], *([1] * num_arbitrary_dims), emb.shape[1])
        else:
            emb_expanded = emb
        """

        # find nearest neighbors
        dist = torch.norm(x_expanded - emb_expanded, 2, 1)
        _, argmin = dist.min(-1) # B,O
        argmin_bk = argmin
        if not share_codebook:
            tmp = torch.arange(O, device=input.device).unsqueeze(0).expand(B,-1) * K
            argmin_bk = argmin.clone()
            argmin += tmp

        shifted_shape = [input.shape[0], *
                         list(input.shape[2:]), input.shape[1]] #B,O,D
        result = emb.t().index_select(0, argmin.view(-1)
                                      ).view(shifted_shape).permute(0, ctx.dims[-1], *ctx.dims[1:-1])#B,D,O

        ctx.save_for_backward(argmin)
        return result.contiguous(), argmin_bk

    @staticmethod
    def backward(ctx, grad_output, argmin=None):
        grad_input = grad_emb = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output

        if ctx.needs_input_grad[1]:
            argmin, = ctx.saved_variables
            latent_indices = torch.arange(ctx.num_emb).type_as(argmin)
            idx_choices = (argmin.view(-1, 1) ==
                           latent_indices.view(1, -1)).type_as(grad_output.data)
            n_idx_choice = idx_choices.sum(0)
            n_idx_choice[n_idx_choice == 0] = 1
            idx_avg_choices = idx_choices / n_idx_choice
            grad_output = grad_output.permute(0, *ctx.dims[2:], 1).contiguous()
            grad_output = grad_output.view(
                ctx.batch_size * ctx.num_latents, ctx.emb_dim)
            grad_emb = torch.sum(grad_output.data.view(-1, ctx.emb_dim, 1) *
                                 idx_avg_choices.view(-1, 1, ctx.num_emb), 0)
        return grad_input, grad_emb, None, None


def nearest_embed(x, emb, share_codebook=True):
    return NearestEmbedFunc().apply(x, emb, share_codebook)


class NearestEmbed(nn.Module):
    def __init__(self, num_embeddings, embeddings_dim, share_codebook=True, init_weight=None):
        super(NearestEmbed, self).__init__()
        if init_weight is not None:
            init_weight = init_weight.reshape(-1).unsqueeze(dim=0).repeat(embeddings_dim, 1)
            self.weight = nn.Parameter(init_weight)
            #self.weight = init_weight
        else:
            self.weight = nn.Parameter(torch.rand(embeddings_dim, num_embeddings))
            #self.weight = torch.rand(embeddings_dim, num_embeddings)
        self.share_codebook = share_codebook

    def forward(self, x, weight_sg=False):
        """Input:
        ---------
        x - (batch_size, emb_size, *)
        """
        return nearest_embed(x, self.weight.detach() if weight_sg else self.weight, self.share_codebook)


# adapted from https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py#L25
# that adapted from https://github.com/deepmind/sonnet


class NearestEmbedEMA(nn.Module):
    def __init__(self, n_emb, emb_dim, decay=0.99, eps=1e-5):
        super(NearestEmbedEMA, self).__init__()
        self.decay = decay
        self.eps = eps
        self.embeddings_dim = emb_dim
        self.n_emb = n_emb
        self.emb_dim = emb_dim
        embed = torch.rand(emb_dim, n_emb)
        self.register_buffer('weight', embed)
        self.register_buffer('cluster_size', torch.zeros(n_emb))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, x):
        """Input:
        ---------
        x - (batch_size, emb_size, *)
        """

        dims = list(range(len(x.size())))
        x_expanded = x.unsqueeze(-1)
        num_arbitrary_dims = len(dims) - 2
        if num_arbitrary_dims:
            emb_expanded = self.weight.view(
                self.emb_dim, *([1] * num_arbitrary_dims), self.n_emb)
        else:
            emb_expanded = self.weight

        # find nearest neighbors
        dist = torch.norm(x_expanded - emb_expanded, 2, 1)
        _, argmin = dist.min(-1)
        shifted_shape = [x.shape[0], *list(x.shape[2:]), x.shape[1]]
        result = self.weight.t().index_select(
            0, argmin.view(-1)).view(shifted_shape).permute(0, dims[-1], *dims[1:-1])

        if self.training:
            latent_indices = torch.arange(self.n_emb).type_as(argmin)
            emb_onehot = (argmin.view(-1, 1) ==
                          latent_indices.view(1, -1)).type_as(x.data)
            n_idx_choice = emb_onehot.sum(0)
            n_idx_choice[n_idx_choice == 0] = 1
            flatten = x.permute(
                1, 0, *dims[-2:]).contiguous().view(x.shape[1], -1)

            self.cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, n_idx_choice
            )
            embed_sum = flatten @ emb_onehot
            self.embed_avg.data.mul_(self.decay).add_(
                1 - self.decay, embed_sum)

            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) /
                (n + self.n_emb * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.weight.data.copy_(embed_normalized)

        return result, argmin