# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Refactored: wav2vec_u.py
#
# Changes from original:
#   1. New RealData class: owns all real phoneme sample preparation.
#      Previously this logic was split between Generator.forward() (token_x
#      creation) and Wav2vec_U.forward() (token_padding_mask, token_y).
#
#   2. Generator.forward(): removed token_x creation. Generator now only
#      produces fake samples from audio segment vectors, as it should.
#
#   3. Wav2vec_U.__init__(): instantiates RealData.
#
#   4. Wav2vec_U.forward(): delegates real sample preparation to RealData
#      instead of handling it inline.
#
# Everything else (architecture, loss computation, Fairseq registration,
# segmentation, config) is unchanged.

from dataclasses import dataclass
from enum import Enum, auto
import math
import numpy as np
from typing import Tuple, List, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

from fairseq import checkpoint_utils, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.modules import (
    SamePad,
    TransposeLast,
)


# Segmentation (unchanged)

class SegmentationType(Enum):
    NONE = auto()
    RANDOM = auto()
    UNIFORM_RANDOM = auto()
    UNIFORM_RANDOM_JOIN = auto()
    JOIN = auto()


@dataclass
class SegmentationConfig(FairseqDataclass):
    type: SegmentationType = SegmentationType.NONE
    subsample_rate: float = 0.25
    mean_pool: bool = True
    mean_pool_join: bool = False
    remove_zeros: bool = False


@dataclass
class Wav2vec_UConfig(FairseqDataclass):
    discriminator_kernel: int = 3
    discriminator_dilation: int = 1
    discriminator_dim: int = 256
    discriminator_causal: bool = True
    discriminator_linear_emb: bool = False
    discriminator_depth: int = 1
    discriminator_max_pool: bool = False
    discriminator_act_after_linear: bool = False
    discriminator_dropout: float = 0.0
    discriminator_spectral_norm: bool = False
    discriminator_weight_norm: bool = False

    generator_kernel: int = 4
    generator_dilation: int = 1
    generator_stride: int = 1
    generator_pad: int = -1
    generator_bias: bool = False
    generator_dropout: float = 0.0
    generator_batch_norm: int = 0
    generator_residual: bool = False

    blank_weight: float = 0
    blank_mode: str = "add"
    blank_is_sil: bool = False
    no_softmax: bool = False

    smoothness_weight: float = 0.0
    smoothing: float = 0.0
    smoothing_one_sided: bool = False
    gradient_penalty: float = 0.0
    probabilistic_grad_penalty_slicing: bool = False
    code_penalty: float = 0.0
    mmi_weight: float = 0.0
    target_dim: int = 64
    target_downsample_rate: int = 2
    gumbel: bool = False
    hard_gumbel: bool = True
    temp: Tuple[float, float, float] = (2, 0.1, 0.99995)
    input_dim: int = 128

    segmentation: SegmentationConfig = SegmentationConfig()


class Segmenter(nn.Module):
    cfg: SegmentationConfig

    def __init__(self, cfg: SegmentationConfig):
        super().__init__()
        self.cfg = cfg
        self.subsample_rate = cfg.subsample_rate

    def pre_segment(self, dense_x, dense_padding_mask):
        return dense_x, dense_padding_mask

    def logit_segment(self, logits, padding_mask):
        return logits, padding_mask


class RandomSegmenter(Segmenter):
    def pre_segment(self, dense_x, dense_padding_mask):
        target_num = math.ceil(dense_x.size(1) * self.subsample_rate)
        ones = torch.ones(dense_x.shape[:-1], device=dense_x.device)
        indices, _ = ones.multinomial(target_num).sort(dim=-1)
        indices_ld = indices.unsqueeze(-1).expand(-1, -1, dense_x.size(-1))
        dense_x = dense_x.gather(1, indices_ld)
        dense_padding_mask = dense_padding_mask.gather(1, index=indices)
        return dense_x, dense_padding_mask


class UniformRandomSegmenter(Segmenter):
    def pre_segment(self, dense_x, dense_padding_mask):
        bsz, tsz, fsz = dense_x.shape

        target_num = math.ceil(tsz * self.subsample_rate)

        rem = tsz % target_num

        if rem > 0:
            dense_x = F.pad(dense_x, [0, 0, 0, target_num - rem])
            dense_padding_mask = F.pad(
                dense_padding_mask, [0, target_num - rem], value=True
            )

        dense_x = dense_x.view(bsz, target_num, -1, fsz)
        dense_padding_mask = dense_padding_mask.view(bsz, target_num, -1)

        if self.cfg.mean_pool:
            dense_x = dense_x.mean(dim=-2)
            dense_padding_mask = dense_padding_mask.all(dim=-1)
        else:
            ones = torch.ones((bsz, dense_x.size(2)), device=dense_x.device)
            indices = ones.multinomial(1)
            indices = indices.unsqueeze(-1).expand(-1, target_num, -1)
            indices_ld = indices.unsqueeze(-1).expand(-1, -1, -1, fsz)
            dense_x = dense_x.gather(2, indices_ld).reshape(bsz, -1, fsz)
            dense_padding_mask = dense_padding_mask.gather(2, index=indices).reshape(
                bsz, -1
            )
        return dense_x, dense_padding_mask


class JoinSegmenter(Segmenter):
    def logit_segment(self, logits, padding_mask):
        preds = logits.argmax(dim=-1)

        if padding_mask.any():
            preds[padding_mask] = -1  # mark pad
        uniques = []

        bsz, tsz, csz = logits.shape

        for p in preds:
            uniques.append(
                p.cpu().unique_consecutive(return_inverse=True, return_counts=True)
            )

        new_tsz = max(u[0].numel() for u in uniques)
        new_logits = logits.new_zeros(bsz, new_tsz, csz)
        new_pad = padding_mask.new_zeros(bsz, new_tsz)

        for b in range(bsz):
            u, idx, c = uniques[b]
            keep = u != -1

            if self.cfg.remove_zeros:
                keep.logical_and_(u != 0)

            if self.training and not self.cfg.mean_pool_join:
                u[0] = 0
                u[1:] = c.cumsum(0)[:-1]
                m = c > 1
                r = torch.rand(m.sum())
                o = (c[m] * r).long()
                u[m] += o
                new_logits[b, : u.numel()] = logits[b, u]
            else:
                new_logits[b].index_add_(
                    dim=0, index=idx.to(new_logits.device), source=logits[b]
                )
                new_logits[b, : c.numel()] /= c.unsqueeze(-1).to(new_logits.device)

            new_sz = keep.sum()
            if not keep.all():
                kept_logits = new_logits[b, : c.numel()][keep]
                new_logits[b, :new_sz] = kept_logits

            if new_sz < new_tsz:
                pad = new_tsz - new_sz
                new_logits[b, -pad:] = 0
                new_pad[b, -pad:] = True

        return new_logits, new_pad


class UniformRandomJoinSegmenter(UniformRandomSegmenter, JoinSegmenter):
    pass


SEGMENT_FACTORY = {
    SegmentationType.NONE: Segmenter,
    SegmentationType.RANDOM: RandomSegmenter,
    SegmentationType.UNIFORM_RANDOM: UniformRandomSegmenter,
    SegmentationType.UNIFORM_RANDOM_JOIN: UniformRandomJoinSegmenter,
    SegmentationType.JOIN: JoinSegmenter,
}


# RealData
#
#Responsibility: given a batch of real phoneme token sequences, produce
# the one-hot vector representations that the Discriminator expects as
# "real" samples.
#
# Previously this logic was split across two places:
#   - token_x creation lived inside Generator.forward() (wrong  Generator
#     should only produce fake samples)
#   - token_padding_mask was computed inline in Wav2vec_U.forward()
#

class RealData:
    """
    Provides real phoneme samples for the Discriminator.

    In the Wav2Vec-U GAN, "real" data is a sequence of one-hot vectors
    derived from phonemized unlabeled text. This class owns the conversion
    from integer token IDs to the one-hot format the Discriminator expects,
    as well as the corresponding padding mask.

    Usage:
        real_data = RealData(pad_index=task.target_dictionary.pad())
        token_x, token_padding_mask = real_data.get_samples(random_label, output_dim)
        real_score = discriminator(token_x, token_padding_mask)
    """

    def __init__(self, pad_index: int):
        """
        Args:
            pad_index: the integer index used for padding in the token vocabulary.
                       Positions with this index are masked out and not scored
                       by the Discriminator.
        """
        self.pad_index = pad_index

    def get_samples(
        self,
        tokens: torch.Tensor,
        output_dim: int,
    ):
        """
        Convert integer phoneme token sequences into one-hot vectors.

        Args:
            tokens: integer token IDs of shape (batch, seq_len).
                    These are real phoneme sequences from the unlabeled text data.
            output_dim: size of the phoneme vocabulary |O|. Each one-hot
                        vector will have this many dimensions.

        Returns:
            token_x: one-hot tensor of shape (batch, seq_len, output_dim).
                     This is what the Discriminator receives as "real" input.
            token_padding_mask: boolean tensor of shape (batch, seq_len).
                                True where the token is a pad token.
        """
        # Padding mask: True at positions that are padding
        token_padding_mask = tokens == self.pad_index

        # Build one-hot vectors.
        # tokens.numel() flattens batch * seq_len into a single dimension,
        # then we scatter a 1 at the correct phoneme index for each position,
        # then reshape back to (batch, seq_len, output_dim).
        token_x = tokens.new_zeros(tokens.numel(), output_dim).float()
        token_x.scatter_(1, tokens.view(-1, 1).long(), 1)
        token_x = token_x.view(tokens.shape + (output_dim,))

        return token_x, token_padding_mask


# Discriminator (unchanged except for docstring clarification)
#
# Responsibility: given a sequence of phoneme distributions (either real
# one-hot vectors from RealData, or soft distributions from Generator),
# output a score per timestep indicating how likely the sequence is real.

class Discriminator(nn.Module):
    def __init__(self, dim, cfg: Wav2vec_UConfig):
        super().__init__()

        inner_dim = cfg.discriminator_dim
        kernel = cfg.discriminator_kernel
        dilation = cfg.discriminator_dilation
        self.max_pool = cfg.discriminator_max_pool

        if cfg.discriminator_causal:
            padding = kernel - 1
        else:
            padding = kernel // 2

        def make_conv(in_d, out_d, k, p=0, has_dilation=True):
            conv = nn.Conv1d(
                in_d,
                out_d,
                kernel_size=k,
                padding=p,
                dilation=dilation if has_dilation else 1,
            )
            if cfg.discriminator_spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            elif cfg.discriminator_weight_norm:
                conv = nn.utils.weight_norm(conv)
            return conv

        inner_net = [
            nn.Sequential(
                make_conv(inner_dim, inner_dim, kernel, padding),
                SamePad(kernel_size=kernel, causal=cfg.discriminator_causal),
                nn.Dropout(cfg.discriminator_dropout),
                nn.GELU(),
            )
            for _ in range(cfg.discriminator_depth - 1)
        ] + [
            make_conv(inner_dim, 1, kernel, padding, has_dilation=False),
            SamePad(kernel_size=kernel, causal=cfg.discriminator_causal),
        ]

        if cfg.discriminator_linear_emb:
            emb_net = [make_conv(dim, inner_dim, 1)]
        else:
            emb_net = [
                make_conv(dim, inner_dim, kernel, padding),
                SamePad(kernel_size=kernel, causal=cfg.discriminator_causal),
            ]

        if cfg.discriminator_act_after_linear:
            emb_net.append(nn.GELU())

        self.net = nn.Sequential(
            *emb_net,
            nn.Dropout(cfg.discriminator_dropout),
            *inner_net,
        )

    def forward(self, x, padding_mask):
        x = x.transpose(1, 2)  # BTC -> BCT
        x = self.net(x)
        x = x.transpose(1, 2)
        x_sz = x.size(1)
        if padding_mask is not None and padding_mask.any() and padding_mask.dim() > 1:
            padding_mask = padding_mask[:, : x.size(1)]
            x[padding_mask] = float("-inf") if self.max_pool else 0
            x_sz = x_sz - padding_mask.sum(dim=-1)
        x = x.squeeze(-1)
        if self.max_pool:
            x, _ = x.max(dim=-1)
        else:
            x = x.sum(dim=-1)
            x = x / x_sz
        return x


# Generator (modified: token_x creation removed)
#
# Responsibility: given audio segment vectors, produce a probability
# distribution over phonemes — the "fake" samples for the Discriminator.
#
# What was removed:
#   The original Generator.forward() created token_x (one-hot real data):
#
#       token_x = None
#       if tokens is not None:
#           token_x = dense_x.new_zeros(tokens.numel(), self.output_dim)
#           token_x.scatter_(1, tokens.view(-1, 1).long(), 1)
#           token_x = token_x.view(tokens.shape + (self.output_dim,))
#
#   This was real data logic inside the Generator — a violation of
#   separation of concerns. It now lives in RealData.get_samples() instead.
#
#   The `tokens` parameter has been removed from forward() since the
#   Generator no longer needs to know about real token sequences at all.

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, cfg: Wav2vec_UConfig):
        super().__init__()

        self.cfg = cfg
        self.output_dim = output_dim
        self.stride = cfg.generator_stride
        self.dropout = nn.Dropout(cfg.generator_dropout)
        self.batch_norm = cfg.generator_batch_norm != 0
        self.residual = cfg.generator_residual

        padding = (
            cfg.generator_kernel // 2 if cfg.generator_pad < 0 else cfg.generator_pad
        )
        self.proj = nn.Sequential(
            TransposeLast(),
            nn.Conv1d(
                input_dim,
                output_dim,
                kernel_size=cfg.generator_kernel,
                stride=cfg.generator_stride,
                dilation=cfg.generator_dilation,
                padding=padding,
                bias=cfg.generator_bias,
            ),
            TransposeLast(),
        )

        if self.batch_norm:
            self.bn = nn.BatchNorm1d(input_dim)
            self.bn.weight.data.fill_(cfg.generator_batch_norm)
        if self.residual:
            self.in_proj = nn.Linear(input_dim, input_dim)

    def forward(self, dense_x, dense_padding_mask):
        """
        Produces fake phoneme distributions from audio segment vectors.

        Args:
            dense_x: audio segment vectors of shape (batch, seq_len, input_dim).
                     These are the PCA-reduced, mean-pooled wav2vec 2.0 features
                     from Section 3.3 of Baevski et al., 2021.
            dense_padding_mask: boolean mask of shape (batch, seq_len).
                                True at padded positions.

        Returns:
            dict with keys:
                dense_x: fake phoneme logits of shape (batch, seq_len, output_dim).
                         These are the Generator's "fake" samples.
                dense_padding_mask: updated padding mask after stride (if any).
                inter_x: intermediate representation if residual=True, else absent.

        Note: token_x (one-hot real data) has been removed from this method.
        Real data is now the responsibility of RealData.get_samples().
        """
        result = {}

        if self.batch_norm:
            dense_x = self.bn_padded_data(dense_x, dense_padding_mask)
        if self.residual:
            inter_x = self.in_proj(self.dropout(dense_x))
            dense_x = dense_x + inter_x
            result["inter_x"] = inter_x

        dense_x = self.dropout(dense_x)
        dense_x = self.proj(dense_x)

        if self.stride > 1:
            dense_padding_mask = dense_padding_mask[:, :: self.stride]

        if dense_padding_mask.size(1) != dense_x.size(1):
            new_padding = dense_padding_mask.new_zeros(dense_x.shape[:-1])
            diff = new_padding.size(1) - dense_padding_mask.size(1)

            if diff > 0:
                new_padding[:, diff:] = dense_padding_mask
            else:
                assert diff < 0
                new_padding = dense_padding_mask[:, :diff]

            dense_padding_mask = new_padding

        result["dense_x"] = dense_x
        result["dense_padding_mask"] = dense_padding_mask

        return result

    def bn_padded_data(self, feature, padding_mask):
        normed_feature = feature.clone()
        normed_feature[~padding_mask] = self.bn(
            feature[~padding_mask].unsqueeze(-1)
        ).squeeze(-1)
        return normed_feature


# Wav2vec_U — the orchestrator (modified to use RealData)
#
# This class coordinates Generator, RealData, and Discriminator.
# Its responsibility is orchestration: deciding whose turn it is to update,
# running the forward pass, and computing losses.
#
# Changes from original:
#   1. __init__: instantiates RealData with the pad index from target_dict
#   2. forward: calls self.real_data.get_samples() instead of handling
#      token_x and token_padding_mask inline
#   3. forward: passes only (dense_x, dense_padding_mask) to Generator
#      since Generator no longer accepts tokens

@register_model("wav2vec_u", dataclass=Wav2vec_UConfig)
class Wav2vec_U(BaseFairseqModel):

    def __init__(self, cfg: Wav2vec_UConfig, target_dict):
        super().__init__()

        self.cfg = cfg
        self.zero_index = target_dict.index("<SIL>") if "<SIL>" in target_dict else 0
        self.smoothness_weight = cfg.smoothness_weight

        output_size = len(target_dict)
        self.pad = target_dict.pad()
        self.eos = target_dict.eos()
        self.smoothing = cfg.smoothing
        self.smoothing_one_sided = cfg.smoothing_one_sided
        self.no_softmax = cfg.no_softmax
        self.gumbel = cfg.gumbel
        self.hard_gumbel = cfg.hard_gumbel
        self.last_acc = None

        self.gradient_penalty = cfg.gradient_penalty
        self.code_penalty = cfg.code_penalty
        self.mmi_weight = cfg.mmi_weight
        self.blank_weight = cfg.blank_weight
        self.blank_mode = cfg.blank_mode
        self.blank_index = target_dict.index("<SIL>") if cfg.blank_is_sil else 0
        assert self.blank_index != target_dict.unk()

        self.discriminator = Discriminator(output_size, cfg)
        for p in self.discriminator.parameters():
            p.param_group = "discriminator"

        self.pca_A = self.pca_b = None
        d = cfg.input_dim

        self.segmenter = SEGMENT_FACTORY[cfg.segmentation.type](cfg.segmentation)

        self.generator = Generator(d, output_size, cfg)
        for p in self.generator.parameters():
            p.param_group = "generator"

        for p in self.segmenter.parameters():
            p.param_group = "generator"

        # NEW: instantiate RealData with the pad index from the target dictionary.
        # RealData owns the conversion of real phoneme tokens to one-hot vectors.
        self.real_data = RealData(pad_index=self.pad)

        self.max_temp, self.min_temp, self.temp_decay = cfg.temp
        self.curr_temp = self.max_temp
        self.update_num = 0

        if self.mmi_weight > 0:
            self.target_downsample_rate = cfg.target_downsample_rate
            self.decoder = nn.Linear(d, cfg.target_dim)
            for p in self.decoder.parameters():
                p.param_group = "generator"

    @classmethod
    def build_model(cls, cfg, task):
        return cls(cfg, task.target_dictionary)

    def calc_gradient_penalty(self, real_data, fake_data):
        b_size = min(real_data.size(0), fake_data.size(0))
        t_size = min(real_data.size(1), fake_data.size(1))

        if self.cfg.probabilistic_grad_penalty_slicing:

            def get_slice(data, dim, target_size):
                size = data.size(dim)
                diff = size - target_size
                if diff <= 0:
                    return data
                start = np.random.randint(0, diff + 1)
                return data.narrow(dim=dim, start=start, length=target_size)

            real_data = get_slice(real_data, 0, b_size)
            real_data = get_slice(real_data, 1, t_size)
            fake_data = get_slice(fake_data, 0, b_size)
            fake_data = get_slice(fake_data, 1, t_size)
        else:
            real_data = real_data[:b_size, :t_size]
            fake_data = fake_data[:b_size, :t_size]

        alpha = torch.rand(real_data.size(0), 1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.to(real_data.device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        disc_interpolates = self.discriminator(interpolates, None)

        gradients = autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=real_data.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradient_penalty = (gradients.norm(2, dim=1) - 1) ** 2
        return gradient_penalty

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.update_num = num_updates
        self.curr_temp = max(
            self.max_temp * self.temp_decay ** num_updates, self.min_temp
        )

    def discrim_step(self, num_updates):
        return num_updates % 2 == 1

    def get_groups_for_update(self, num_updates):
        return "discriminator" if self.discrim_step(num_updates) else "generator"

    def get_logits(
        self,
        net_output: Optional[Dict[str, List[Optional[torch.Tensor]]]],
        normalize: bool = False,
    ):
        logits = net_output["logits"]

        if self.blank_weight != 0:
            if self.blank_mode == "add":
                logits[..., self.blank_index] += self.blank_weight
            elif self.blank_mode == "set":
                logits[..., self.blank_index] = self.blank_weight
            else:
                raise Exception(f"invalid blank mode {self.blank_mode}")

        padding = net_output["padding_mask"]
        if padding.any():
            logits[padding] = float("-inf")
            logits[padding][..., self.blank_index] = float("inf")

        if normalize:
            logits = utils.log_softmax(logits.float(), dim=-1)

        return logits.transpose(0, 1)

    def get_normalized_probs(
        self,
        net_output: Tuple[
            torch.Tensor, Optional[Dict[str, List[Optional[torch.Tensor]]]]
        ],
        log_probs: bool,
        sample: Optional[Dict[str, torch.Tensor]] = None,
    ):
        logits = self.get_logits(net_output)
        probs = super().get_normalized_probs(logits, log_probs, sample)
        probs = probs.transpose(0, 1)
        return probs

    def normalize(self, dense_x):
        bsz, tsz, csz = dense_x.shape

        if dense_x.numel() == 0:
            raise Exception(dense_x.shape)
        _, k = dense_x.max(-1)
        hard_x = (
            dense_x.new_zeros(bsz * tsz, csz)
            .scatter_(-1, k.view(-1, 1), 1.0)
            .view(-1, csz)
        )
        hard_probs = torch.mean(hard_x.float(), dim=0)
        code_perplexity = torch.exp(
            -torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)
        )

        avg_probs = torch.softmax(dense_x.reshape(-1, csz).float(), dim=-1).mean(dim=0)
        prob_perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1)
        )

        if not self.no_softmax:
            if self.training and self.gumbel:
                dense_x = F.gumbel_softmax(
                    dense_x.float(), tau=self.curr_temp, hard=self.hard_gumbel
                ).type_as(dense_x)
            else:
                dense_x = dense_x.softmax(-1)

        return dense_x, code_perplexity, prob_perplexity

    def forward(
        self,
        features,
        padding_mask,
        random_label=None,
        dense_x_only=False,
        segment=True,
        aux_target=None,
    ):
        if segment:
            features, padding_mask = self.segmenter.pre_segment(features, padding_mask)

        orig_size = features.size(0) * features.size(1) - padding_mask.sum()

        # Changed: Generator.forward() no longer accepts tokens (random_label).
        # It only takes audio features and produces fake phoneme distributions.
        gen_result = self.generator(features, padding_mask)

        orig_dense_x = gen_result["dense_x"]
        orig_dense_padding_mask = gen_result["dense_padding_mask"]

        if segment:
            dense_x, dense_padding_mask = self.segmenter.logit_segment(
                orig_dense_x, orig_dense_padding_mask
            )
        else:
            dense_x = orig_dense_x
            dense_padding_mask = orig_dense_padding_mask

        dense_logits = dense_x
        prob_perplexity = None
        code_perplexity = None

        if not (self.no_softmax and dense_x_only):
            dense_x, code_perplexity, prob_perplexity = self.normalize(dense_logits)

        if dense_x_only or self.discriminator is None:
            return {
                "logits": dense_x,
                "padding_mask": dense_padding_mask,
            }

        # Changed: real sample preparation now delegated to RealData.
        # Previously: token_padding_mask and token_x were computed inline here.
        # Now: RealData.get_samples() owns this responsibility cleanly.
        token_x, token_padding_mask = self.real_data.get_samples(
            tokens=random_label,
            output_dim=self.generator.output_dim,
        )

        # Discriminator scores both fake (from Generator) and real (from RealData)
        dense_y = self.discriminator(dense_x, dense_padding_mask)
        token_y = self.discriminator(token_x, token_padding_mask)

        sample_size = features.size(0)
        d_step = self.discrim_step(self.update_num)

        fake_smooth = self.smoothing
        real_smooth = self.smoothing
        if self.smoothing_one_sided:
            fake_smooth = 0

        zero_loss = None
        smoothness_loss = None
        code_pen = None
        mmi_loss = None

        if d_step:
            # Discriminator update:
            # real samples (token_y) should score high → target 1
            # fake samples (dense_y) should score low → target 0
            loss_dense = F.binary_cross_entropy_with_logits(
                dense_y,
                dense_y.new_ones(dense_y.shape) - fake_smooth,
                reduction="sum",
            )
            loss_token = F.binary_cross_entropy_with_logits(
                token_y,
                token_y.new_zeros(token_y.shape) + real_smooth,
                reduction="sum",
            )
            if self.training and self.gradient_penalty > 0:
                grad_pen = self.calc_gradient_penalty(token_x, dense_x)
                grad_pen = grad_pen.sum() * self.gradient_penalty
            else:
                grad_pen = None
        else:
            # Generator update:
            # Generator wants fake samples (dense_y) to score high → target 1
            grad_pen = None
            loss_token = None
            loss_dense = F.binary_cross_entropy_with_logits(
                dense_y,
                dense_y.new_zeros(dense_y.shape) + fake_smooth,
                reduction="sum",
            )
            num_vars = dense_x.size(-1)
            if prob_perplexity is not None:
                code_pen = (num_vars - prob_perplexity) / num_vars
                code_pen = code_pen * sample_size * self.code_penalty

            if self.smoothness_weight > 0:
                smoothness_loss = F.mse_loss(
                    dense_logits[:, :-1], dense_logits[:, 1:], reduction="none"
                )
                smoothness_loss[dense_padding_mask[:, 1:]] = 0
                smoothness_loss = (
                    smoothness_loss.mean() * sample_size * self.smoothness_weight
                )

            if (self.mmi_weight > 0) and (aux_target is not None):
                inter_x = self.decoder(gen_result["inter_x"])
                if self.target_downsample_rate > 1:
                    aux_target = aux_target[:, :: self.target_downsample_rate]
                max_t_len = min(aux_target.shape[1], inter_x.shape[1])
                mmi_loss = F.cross_entropy(
                    inter_x[:, :max_t_len].transpose(1, 2),
                    aux_target[:, :max_t_len],
                    ignore_index=-1,
                    reduction="none",
                )
                mmi_loss = mmi_loss.mean() * mmi_loss.shape[0] * self.mmi_weight

        result = {
            "losses": {
                "grad_pen": grad_pen,
                "code_pen": code_pen,
                "smoothness": smoothness_loss,
                "mmi": mmi_loss,
            },
            "temp": self.curr_temp,
            "code_ppl": code_perplexity,
            "prob_ppl": prob_perplexity,
            "d_steps": int(d_step),
            "sample_size": sample_size,
        }

        suff = "_d" if d_step else "_g"
        result["losses"]["dense" + suff] = loss_dense
        result["losses"]["token" + suff] = loss_token

        return result