"""Contains base attention modules."""

import math

import torch
import torch.nn as nn

class SaveAttMaps(nn.Module):
    def __init__(self):
        super().__init__()
        self.curr_att_maps = None
        self.Q = None
        self.K = None
        self.V = None
        self.out = None
        self.out_pre_res = None
        
        self.processed_att = None

    def forward(self, X, Q, K, V):
        self.curr_att_maps = nn.Parameter(X)
        self.Q = nn.Parameter(Q)
        self.K = nn.Parameter(K)
        self.V = nn.Parameter(V)

        return X


class MAB(nn.Module):
    """Multi-head Attention Block.

    Based on Set Transformer implementation
    (Lee et al. 2019, https://github.com/juho-lee/set_transformer).
    """
    def __init__(
            self, dim_Q, dim_KV, dim_emb, dim_out, c, mask=False):
        """

        Inputs have shape (B_A, N_A, F_A), where
        * `B_A` is a batch dimension, along we parallelise computation,
        * `N_A` is the number of samples in each batch, along which we perform
        attention, and
        * `F_A` is dimension of the embedding at input
            * `F_A` is `dim_Q` for query matrix
            * `F_A` is `dim_KV` for key and value matrix.

        Q, K, and V then all get embedded to `dim_emb`.
        `dim_out` is the output dimensionality of the MAB which has shape
        (B_A, N_A, dim_out), this can be different from `dim_KV` due to
        the head_mixing.

        This naming scheme is inherited from set-transformer paper.
        """
        super(MAB, self).__init__()
        mix_heads = False
        num_heads = c.model_num_heads
        sep_res_embed = True
        ln = True
        self.debug_mode = False
        rff_depth = 1
        self.att_score_norm = 'softmax'
        self.pre_layer_norm = True
        self.viz_att_maps = c.viz_att_maps
        self.model_ablate_rff = False
        self.mask = mask
        # self.do_mask_batch = c.do_mask_batch
        if self.viz_att_maps:
            self.save_att_maps = SaveAttMaps()
        
        self.null_KV = c.null_KV
        if self.null_KV:
            self.null_k = nn.Parameter(torch.zeros(1, 1, dim_KV))
        
        self.topk = c.topk
        self.norm_after_topk = c.norm_after_topk
        self.softmax_temp = c.softmax_temp

        if dim_out is None:
            dim_out = dim_emb
        elif (dim_out is not None) and (mix_heads is None):
            print('Warning: dim_out transformation does not apply.')
            dim_out = dim_emb

        self.num_heads = num_heads
        self.dim_KV = dim_KV
        self.dim_split = dim_emb // num_heads
        self.fc_q = nn.Linear(dim_Q, dim_emb)
        self.fc_k = nn.Linear(dim_KV, dim_emb)
        self.fc_v = nn.Linear(dim_KV, dim_emb)
        self.fc_mix_heads = nn.Linear(dim_emb, dim_out) if mix_heads else None
        if sep_res_embed:
            if dim_out != dim_Q:
                print('Warning: dim_out != dim_Q. Using Linear transform to align dimensions.')
                self.fc_res = nn.Linear(dim_Q, dim_out)
            else:
                print('Info: dim_out == dim_Q. Using Identity transform')
                self.fc_res = nn.Identity()
        else:
            self.fc_res = None
        if ln:
            if self.pre_layer_norm:  # Applied to X
                self.ln0 = nn.LayerNorm(dim_Q, eps=c.model_layer_norm_eps)
                self.ln0KV = nn.LayerNorm(dim_KV, eps=c.model_layer_norm_eps)
            else:  # Applied after MHA and residual
                self.ln0 = nn.LayerNorm(dim_out, eps=c.model_layer_norm_eps)

            self.ln1 = nn.LayerNorm(dim_out, eps=c.model_layer_norm_eps)
        else:
            self.ln0 = None
            self.ln1 = None

        self.hidden_dropout = (
            nn.Dropout(p=c.model_hidden_dropout_prob)
            if c.model_hidden_dropout_prob else None)

        # self.att_scores_dropout = (
        #     nn.Dropout(p=c.model_att_score_dropout_prob)
        #     if c.model_att_score_dropout_prob else None)
        self.att_scores_dropout = None
        
        self.init_rff(dim_out, rff_depth)

    def init_rff(self, dim_out, rff_depth):
        # Linear layer with 4 times expansion factor as in 'Attention is
        # all you need'!
        self.rff = [nn.Linear(dim_out, 4 * dim_out), nn.GELU()]

        if self.hidden_dropout is not None:
            self.rff.append(self.hidden_dropout)

        for i in range(rff_depth - 1):
            self.rff += [nn.Linear(4 * dim_out, 4 * dim_out), nn.GELU()]

            if self.hidden_dropout is not None:
                self.rff.append(self.hidden_dropout)

        self.rff += [nn.Linear(4 * dim_out, dim_out)]

        if self.hidden_dropout is not None:
            self.rff.append(self.hidden_dropout)

        self.rff = nn.Sequential(*self.rff)
        
    def forward(self, X, Y, mask=None):
        if self.pre_layer_norm and self.ln0 is not None:
            X_multihead = self.ln0(X)
            Y = self.ln0KV(Y)
        else:
            X_multihead = X

        Q = self.fc_q(X_multihead)

        if self.fc_res is None:
            X_res = Q
        else:
            X_res = self.fc_res(X)  # Separate embedding for residual

        K = self.fc_k(Y)
        V = self.fc_v(Y)
        
        if self.debug_mode:
            print('Q', Q.shape)
            print('K', K.shape)
            print('V', V.shape)    
    
        if self.null_KV:
            attatch_k_zero = self.null_k.expand(K.shape[0], -1, -1)
            K = torch.cat([K, attatch_k_zero], dim=-2)
            attatch_v_zero = torch.zeros((V.shape[0], 1, V.shape[-1]), device=V.device)
            V = torch.cat([V, attatch_v_zero], dim=-2)

        Q_ = torch.cat(Q.split(self.dim_split, 2), 0)
        K_ = torch.cat(K.split(self.dim_split, 2), 0)
        V_ = torch.cat(V.split(self.dim_split, 2), 0)
        
        if self.debug_mode:
            print('Q_', Q_.shape)
            print('K_', K_.shape)
            print('V_', V_.shape)
            
        # TODO: track issue at
        # https://github.com/juho-lee/set_transformer/issues/8
        # A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        A = torch.einsum('ijl,ikl->ijk', Q_, K_)
        
        if self.debug_mode:
            print('A', A.shape)

        if mask is not None:
            A = A * mask

        if self.att_score_norm == 'softmax':
            if self.softmax_temp != 1.0:
                A = torch.softmax((A / math.sqrt(self.dim_KV))* self.softmax_temp, 2)
            else:
                A = torch.softmax(A / math.sqrt(self.dim_KV), 2)
        elif self.att_score_norm == 'constant':
            A = A / self.dim_split
        else:
            raise NotImplementedError

        if self.viz_att_maps:
            print("viz raw att maps")
            self.save_att_maps.processed_att = A.detach().cpu()

        # Attention scores dropout is applied to the N x N_v matrix of
        # attention scores.
        # Hence, it drops out entire rows/cols to attend to.
        # This follows Vaswani et al. 2017 (original Transformer paper).

        if self.att_scores_dropout is not None:
            A = self.att_scores_dropout(A)
        
        if self.topk is not None and self.topk < A.shape[2]:
            topk_ind = torch.topk(A, dim = 2, k = self.topk)
            k_mask = torch.zeros(A.size(), device=A.device)
            k_mask = k_mask.scatter(2, topk_ind.indices, 1)
            A = A * k_mask
            if self.norm_after_topk:
                A = A / torch.sum(A, dim = 2, keepdim = True)

        if self.viz_att_maps:
            print("viz processed att maps")
            A = self.save_att_maps(A, Q_, K_, V_)

        multihead = A.bmm(V_)
        
        if self.debug_mode:
            print('multihead_before', multihead.shape)
        
        multihead = torch.cat(multihead.split(Q.size(0), 0), 2)

        if self.debug_mode:
            print('multihead_after', multihead.shape)

        # Add mixing of heads in hidden dim.

        if self.fc_mix_heads is not None:
            H = self.fc_mix_heads(multihead)
        else:
            H = multihead

        # Follow Vaswani et al. 2017 in applying dropout prior to
        # residual and LayerNorm
        if self.hidden_dropout is not None:
            H = self.hidden_dropout(H)

        # True to the paper would be to replace
        # self.fc_mix_heads = nn.Linear(dim_V, dim_Q)
        # and Q_out = X
        # Then, the output dim is equal to input dim, just like it's written
        # in the paper. We should definitely check if that boosts performance.
        # This will require changes to downstream structure (since downstream
        # blocks expect input_dim=dim_V and not dim_Q)

        # Residual connection
        # Q_out = X_res
        H = H + X_res
        # ResH

        # Post Layer-Norm, as in SetTransformer and BERT.
        if not self.pre_layer_norm and self.ln0 is not None:
            H = self.ln0(H)

        if self.pre_layer_norm and self.ln1 is not None:
            H_rff = self.ln1(H)
        else:
            H_rff = H

        if self.model_ablate_rff:
            expanded_linear_H = H_rff
        else:
            # Apply row-wise feed forward network
            expanded_linear_H = self.rff(H_rff)

        # Residual connection
        expanded_linear_H = H + expanded_linear_H

        if not self.pre_layer_norm and self.ln1 is not None:
            expanded_linear_H = self.ln1(expanded_linear_H)

        if self.viz_att_maps:
            self.save_att_maps.out = nn.Parameter(expanded_linear_H)
            self.save_att_maps.out_pre_res = nn.Parameter(H)

        return expanded_linear_H


class MHSA(nn.Module):
    """
    Multi-head Self-Attention Block.

    Based on implementation from Set Transformer (Lee et al. 2019,
    https://github.com/juho-lee/set_transformer).
    Alterations detailed in MAB method.
    """
    has_inducing_points = False

    def __init__(self, dim_in, dim_emb, dim_out, c):
        super(MHSA, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_emb, dim_out, c)
    def forward(self, Q, KV, mask=None):
        return self.mab(Q, KV, mask=mask)
    # def forward(self, X, mask=None):
    #     return self.mab(X, X, mask=mask)
