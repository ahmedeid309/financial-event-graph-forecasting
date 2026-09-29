import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
import numpy as np


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super(EnEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars


class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, graph_cross=None, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(
                x,
                cross,
                graph_cross=graph_cross,
                x_mask=x_mask,
                cross_mask=cross_mask,
                tau=tau,
                delta=delta,
            )

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, graph_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu", graph_gate_init=0.0, graph_gate_enabled=True):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.graph_attention = graph_attention
        # ReZero-style residual gate. At exactly zero, attaching the graph
        # branch cannot perturb the price-only forecast, but the gate itself
        # still receives a gradient and can learn to open in either direction.
        self.graph_gate = nn.Parameter(torch.tensor(float(graph_gate_init)))
        # Ablation switch. When False the graph residual is added at full
        # strength with no gate, so the branch is live from step zero.
        self.graph_gate_enabled = bool(graph_gate_enabled)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, graph_cross=None, x_mask=None, cross_mask=None, tau=None, delta=None):
        B, _, D = cross.shape
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))
        x_glb_attn = self.dropout(self.cross_attention(
            x_glb, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])
        x_glb_attn = torch.reshape(x_glb_attn,
                                   (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])).unsqueeze(1)
        x_glb = x_glb_ori + x_glb_attn
        if graph_cross is not None:
            graph_glb_attn = self.dropout(self.graph_attention(
                torch.reshape(x_glb_ori, (x_glb_ori.shape[0], -1, D)),
                graph_cross,
                graph_cross,
                attn_mask=None,
                tau=tau,
                delta=delta,
            )[0])
            graph_glb_attn = torch.reshape(
                graph_glb_attn,
                (graph_glb_attn.shape[0] * graph_glb_attn.shape[1], graph_glb_attn.shape[2]),
            ).unsqueeze(1)
            if self.graph_gate_enabled:
                x_glb = x_glb + torch.tanh(self.graph_gate) * graph_glb_attn
            else:
                x_glb = x_glb + graph_glb_attn
        x_glb = self.norm2(x_glb)

        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.features = configs.features
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.patch_len = configs.patch_len
        self.patch_num = int(configs.seq_len // configs.patch_len)
        self.n_vars = 1 if configs.features == 'MS' else configs.enc_in
        self.graph_feature_count = int(getattr(configs, 'graph_feature_count', 0) or 0)
        self.graph_adapter_dim = int(getattr(configs, 'graph_adapter_dim', 4) or 4)
        self.graph_gate_init = float(getattr(configs, 'graph_gate_init', 0.0) or 0.0)
        self.graph_gate_enabled = not bool(getattr(configs, 'no_graph_gate', False))
        # 'attention' = graph gets its own cross-attention (production).
        # 'concat'    = graph factors join the ordinary exogenous variables,
        #              so the gate and the bottleneck can be measured without
        #              the separate attention pathway.
        self.graph_fusion_mode = str(getattr(configs, 'graph_fusion_mode', 'attention'))
        if self.graph_feature_count < 0:
            raise ValueError('graph_feature_count must be non-negative')
        if self.graph_feature_count > 0 and configs.features != 'MS':
            raise ValueError('The gated graph adapter currently requires --features MS')
        if self.graph_feature_count > 0 and self.graph_adapter_dim <= 0:
            raise ValueError('graph_adapter_dim must be positive when graph features are enabled')
        # Embedding
        self.en_embedding = EnEmbedding(self.n_vars, configs.d_model, self.patch_len, configs.dropout)

        self.ex_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                   configs.dropout)

        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    graph_gate_init=self.graph_gate_init,
                    graph_gate_enabled=self.graph_gate_enabled,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.head_nf = configs.d_model * (self.patch_num + 1)
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                head_dropout=configs.dropout)

        # These modules are deliberately created after the complete price
        # pathway. Thus paired runs with the same seed initialise every shared
        # price parameter identically, whether the adapter is enabled or not.
        self.concat_gate = nn.Parameter(torch.tensor(float(self.graph_gate_init)))
        if self.graph_feature_count > 0:
            # LayerNorm normalises ACROSS the news-feature dimension. With a
            # single feature its output is constant for every input -- variance
            # over one element is zero -- so a one-dimensional representation
            # would be destroyed before it reached the adapter. Every news
            # representation is already standardised on training rows when it is
            # exported, so the normalisation is a safeguard rather than a
            # requirement, and it is skipped where it cannot be computed.
            # Widths above one are unaffected: the 64- and 1024-dimensional arms
            # construct exactly the module they did before.
            normalizer = (
                nn.LayerNorm(self.graph_feature_count)
                if self.graph_feature_count > 1
                else nn.Identity()
            )
            self.graph_adapter = nn.Sequential(
                normalizer,
                nn.Linear(self.graph_feature_count, self.graph_adapter_dim),
                nn.GELU(),
                nn.Dropout(configs.dropout),
            )
            self.graph_embedding = DataEmbedding_inverted(
                configs.seq_len,
                configs.d_model,
                configs.embed,
                configs.freq,
                configs.dropout,
            )
        else:
            self.graph_adapter = None
            self.graph_embedding = None

    def freeze_price_pathway(self):
        """Freeze the baseline and leave only the graph residual trainable."""

        for parameter in self.parameters():
            parameter.requires_grad = False
        if self.graph_adapter is None or self.graph_embedding is None:
            raise ValueError('Cannot freeze for graph fusion without graph features')
        for parameter in self.graph_adapter.parameters():
            parameter.requires_grad = True
        if self.graph_fusion_mode == 'concat':
            if self.graph_gate_enabled:
                self.concat_gate.requires_grad = True
            return
        for parameter in self.graph_embedding.parameters():
            parameter.requires_grad = True
        for layer in self.encoder.layers:
            for parameter in layer.graph_attention.parameters():
                parameter.requires_grad = True
            if layer.graph_gate_enabled:
                layer.graph_gate.requires_grad = True

    def graph_gate_values(self):
        if self.graph_fusion_mode == 'concat':
            return [float(torch.tanh(self.concat_gate).detach().cpu())
                    if self.graph_gate_enabled else 1.0]
        return [float(torch.tanh(layer.graph_gate).detach().cpu())
                if layer.graph_gate_enabled else 1.0
                for layer in self.encoder.layers]

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape

        if self.graph_feature_count > 0:
            expected_minimum = self.graph_feature_count + 1
            if N < expected_minimum:
                raise ValueError(
                    f'Input has {N} features but graph_feature_count={self.graph_feature_count} '
                    'requires the graph block plus the final target.'
                )
            graph_start = N - self.graph_feature_count - 1
            price_exogenous = x_enc[:, :, :graph_start]
            graph_features = x_enc[:, :, graph_start:-1]
            graph_factors = self.graph_adapter(graph_features)
            if self.graph_fusion_mode == 'concat':
                if self.graph_gate_enabled:
                    graph_factors = torch.tanh(self.concat_gate) * graph_factors
                price_exogenous = torch.cat([price_exogenous, graph_factors], dim=-1)
                graph_embed = None
            else:
                graph_embed = self.graph_embedding(graph_factors, None)
        else:
            price_exogenous = x_enc[:, :, :-1]
            graph_embed = None

        en_embed, n_vars = self.en_embedding(x_enc[:, :, -1].unsqueeze(-1).permute(0, 2, 1))
        ex_embed = self.ex_embedding(price_exogenous, x_mark_enc)

        enc_out = self.encoder(en_embed, ex_embed, graph_cross=graph_embed)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out


    def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape

        en_embed, n_vars = self.en_embedding(x_enc.permute(0, 2, 1))
        ex_embed = self.ex_embedding(x_enc, x_mark_enc)

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            if self.features == 'M':
                dec_out = self.forecast_multi(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len:, :]  # [B, L, D]
            else:
                dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        else:
            return None
