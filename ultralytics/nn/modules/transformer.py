# Ultralytics 馃殌 AGPL-3.0 License - https://ultralytics.com/license
"""Transformer modules."""

from __future__ import annotations

import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils.torch_utils import TORCH_1_11

from .conv import Conv
from .utils import _get_clones, inverse_sigmoid, multi_scale_deformable_attn_pytorch

__all__ = (
    "AIFI",
    "MLP",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "LayerNorm2d",
    "MLPBlock",
    "MSDeformAttn",
    "WaveletOffsetMSDeformAttn",
    "TransformerBlock",
    "TransformerEncoderLayer",
    "TransformerLayer",
    "MSDeformFusionBlock",
    "Wavelet2D",
    "WaveletGuidedMSDeformFusionBlock",
)


class TransformerEncoderLayer(nn.Module):
    """A single layer of the transformer encoder.

    This class implements a standard transformer encoder layer with multi-head attention and feedforward network,
    supporting both pre-normalization and post-normalization configurations.

    Attributes:
        ma (nn.MultiheadAttention): Multi-head attention module.
        fc1 (nn.Linear): First linear layer in the feedforward network.
        fc2 (nn.Linear): Second linear layer in the feedforward network.
        norm1 (nn.LayerNorm): Layer normalization after attention.
        norm2 (nn.LayerNorm): Layer normalization after feedforward network.
        dropout (nn.Dropout): Dropout layer for the feedforward network.
        dropout1 (nn.Dropout): Dropout layer after attention.
        dropout2 (nn.Dropout): Dropout layer after feedforward network.
        act (nn.Module): Activation function.
        normalize_before (bool): Whether to apply normalization before attention and feedforward.
    """

    def __init__(
        self,
        c1: int,
        cm: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.0,
        act: nn.Module = nn.GELU(),
        normalize_before: bool = False,
    ):
        """Initialize the TransformerEncoderLayer with specified parameters.

        Args:
            c1 (int): Input dimension.
            cm (int): Hidden dimension in the feedforward network.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability.
            act (nn.Module): Activation function.
            normalize_before (bool): Whether to apply normalization before attention and feedforward.
        """
        super().__init__()
        from ...utils.torch_utils import TORCH_1_9

        if not TORCH_1_9:
            raise ModuleNotFoundError(
                "TransformerEncoderLayer() requires torch>=1.9 to use nn.MultiheadAttention(batch_first=True)."
            )
        self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)

        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = act
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos: torch.Tensor | None = None) -> torch.Tensor:
        """Add position embeddings to the tensor if provided."""
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Perform forward pass with post-normalization.

        Args:
            src (torch.Tensor): Input tensor.
            src_mask (torch.Tensor, optional): Mask for the src sequence.
            src_key_padding_mask (torch.Tensor, optional): Mask for the src keys per batch.
            pos (torch.Tensor, optional): Positional encoding.

        Returns:
            (torch.Tensor): Output tensor after attention and feedforward.
        """
        q = k = self.with_pos_embed(src, pos)
        src2 = self.ma(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def forward_pre(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Perform forward pass with pre-normalization.

        Args:
            src (torch.Tensor): Input tensor.
            src_mask (torch.Tensor, optional): Mask for the src sequence.
            src_key_padding_mask (torch.Tensor, optional): Mask for the src keys per batch.
            pos (torch.Tensor, optional): Positional encoding.

        Returns:
            (torch.Tensor): Output tensor after attention and feedforward.
        """
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.ma(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src2))))
        return src + self.dropout2(src2)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward propagate the input through the encoder module.

        Args:
            src (torch.Tensor): Input tensor.
            src_mask (torch.Tensor, optional): Mask for the src sequence.
            src_key_padding_mask (torch.Tensor, optional): Mask for the src keys per batch.
            pos (torch.Tensor, optional): Positional encoding.

        Returns:
            (torch.Tensor): Output tensor after transformer encoder layer.
        """
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class AIFI(TransformerEncoderLayer):
    """AIFI transformer layer for 2D data with positional embeddings.

    This class extends TransformerEncoderLayer to work with 2D feature maps by adding 2D sine-cosine positional
    embeddings and handling the spatial dimensions appropriately.
    """

    def __init__(
        self,
        c1: int,
        cm: int = 2048,
        num_heads: int = 8,
        dropout: float = 0,
        act: nn.Module = nn.GELU(),
        normalize_before: bool = False,
    ):
        """Initialize the AIFI instance with specified parameters.

        Args:
            c1 (int): Input dimension.
            cm (int): Hidden dimension in the feedforward network.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability.
            act (nn.Module): Activation function.
            normalize_before (bool): Whether to apply normalization before attention and feedforward.
        """
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the AIFI transformer layer.

        Args:
            x (torch.Tensor): Input tensor with shape [B, C, H, W].

        Returns:
            (torch.Tensor): Output tensor with shape [B, C, H, W].
        """
        c, h, w = x.shape[1:]
        pos_embed = self.build_2d_sincos_position_embedding(w, h, c)
        # Flatten [B, C, H, W] to [B, HxW, C]
        x = super().forward(x.flatten(2).permute(0, 2, 1), pos=pos_embed.to(device=x.device, dtype=x.dtype))
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()

    @staticmethod
    def build_2d_sincos_position_embedding(
        w: int, h: int, embed_dim: int = 256, temperature: float = 10000.0
    ) -> torch.Tensor:
        """Build 2D sine-cosine position embedding.

        Args:
            w (int): Width of the feature map.
            h (int): Height of the feature map.
            embed_dim (int): Embedding dimension.
            temperature (float): Temperature for the sine/cosine functions.

        Returns:
            (torch.Tensor): Position embedding with shape [1, embed_dim, h*w].
        """
        assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij") if TORCH_1_11 else torch.meshgrid(grid_w, grid_h)
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)[None]


class TransformerLayer(nn.Module):
    """Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)."""

    def __init__(self, c: int, num_heads: int):
        """Initialize a self-attention mechanism using linear transformations and multi-head attention.

        Args:
            c (int): Input and output channel dimension.
            num_heads (int): Number of attention heads.
        """
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a transformer block to the input x and return the output.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after transformer layer.
        """
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        return self.fc2(self.fc1(x)) + x


class TransformerBlock(nn.Module):
    """Vision Transformer block based on https://arxiv.org/abs/2010.11929.

    This class implements a complete transformer block with optional convolution layer for channel adjustment, learnable
    position embedding, and multiple transformer layers.

    Attributes:
        conv (Conv, optional): Convolution layer if input and output channels differ.
        linear (nn.Linear): Learnable position embedding.
        tr (nn.Sequential): Sequential container of transformer layers.
        c2 (int): Output channel dimension.
    """

    def __init__(self, c1: int, c2: int, num_heads: int, num_layers: int):
        """Initialize a Transformer module with position embedding and specified number of heads and layers.

        Args:
            c1 (int): Input channel dimension.
            c2 (int): Output channel dimension.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer layers.
        """
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagate the input through the transformer block.

        Args:
            x (torch.Tensor): Input tensor with shape [b, c1, w, h].

        Returns:
            (torch.Tensor): Output tensor with shape [b, c2, w, h].
        """
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class MLPBlock(nn.Module):
    """A single block of a multi-layer perceptron."""

    def __init__(self, embedding_dim: int, mlp_dim: int, act=nn.GELU):
        """Initialize the MLPBlock with specified embedding dimension, MLP dimension, and activation function.

        Args:
            embedding_dim (int): Input and output dimension.
            mlp_dim (int): Hidden dimension.
            act (nn.Module): Activation function.
        """
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the MLPBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after MLP block.
        """
        return self.lin2(self.act(self.lin1(x)))


class MLP(nn.Module):
    """A simple multi-layer perceptron (also called FFN).

    This class implements a configurable MLP with multiple linear layers, activation functions, and optional sigmoid
    output activation.

    Attributes:
        num_layers (int): Number of layers in the MLP.
        layers (nn.ModuleList): List of linear layers.
        sigmoid (bool): Whether to apply sigmoid to the output.
        act (nn.Module): Activation function.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, act=nn.ReLU, sigmoid: bool = False
    ):
        """Initialize the MLP with specified input, hidden, output dimensions and number of layers.

        Args:
            input_dim (int): Input dimension.
            hidden_dim (int): Hidden dimension.
            output_dim (int): Output dimension.
            num_layers (int): Number of layers.
            act (nn.Module): Activation function.
            sigmoid (bool): Whether to apply sigmoid to the output.
        """
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim, *h], [*h, output_dim]))
        self.sigmoid = sigmoid
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the entire MLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after MLP.
        """
        for i, layer in enumerate(self.layers):
            x = getattr(self, "act", nn.ReLU())(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x.sigmoid() if getattr(self, "sigmoid", False) else x


class LayerNorm2d(nn.Module):
    """2D Layer Normalization module inspired by Detectron2 and ConvNeXt implementations.

    This class implements layer normalization for 2D feature maps, normalizing across the channel dimension while
    preserving spatial dimensions.

    Attributes:
        weight (nn.Parameter): Learnable scale parameter.
        bias (nn.Parameter): Learnable bias parameter.
        eps (float): Small constant for numerical stability.

    References:
        https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py
        https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        """Initialize LayerNorm2d with the given parameters.

        Args:
            num_channels (int): Number of channels in the input.
            eps (float): Small constant for numerical stability.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass for 2D layer normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Normalized output tensor.
        """
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class MSDeformAttn(nn.Module):
    """Multiscale Deformable Attention Module based on Deformable-DETR and PaddleDetection implementations.

    This module implements multiscale deformable attention that can attend to features at multiple scales with learnable
    sampling locations and attention weights.

    Attributes:
        im2col_step (int): Step size for im2col operations.
        d_model (int): Model dimension.
        n_levels (int): Number of feature levels.
        n_heads (int): Number of attention heads.
        n_points (int): Number of sampling points per attention head per feature level.
        sampling_offsets (nn.Linear): Linear layer for generating sampling offsets.
        attention_weights (nn.Linear): Linear layer for generating attention weights.
        value_proj (nn.Linear): Linear layer for projecting values.
        output_proj (nn.Linear): Linear layer for projecting output.

    References:
        https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py
    """

    def __init__(self, d_model: int = 256, n_levels: int = 4, n_heads: int = 8, n_points: int = 4):
        """Initialize MSDeformAttn with the given parameters.

        Args:
            d_model (int): Model dimension.
            n_levels (int): Number of feature levels.
            n_heads (int): Number of attention heads.
            n_points (int): Number of sampling points per attention head per feature level.
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, but got {d_model} and {n_heads}")
        _d_per_head = d_model // n_heads
        # Better to set _d_per_head to a power of 2 which is more efficient in a CUDA implementation
        assert _d_per_head * n_heads == d_model, "`d_model` must be divisible by `n_heads`"

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        """Reset module parameters."""
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

        # Reduce learning rate for sampling offsets by scaling gradients
        self.sampling_offsets.weight.register_hook(lambda grad: grad * 0.1)
        self.sampling_offsets.bias.register_hook(lambda grad: grad * 0.1)

    def forward(
        self,
        query: torch.Tensor,
        refer_bbox: torch.Tensor,
        value: torch.Tensor,
        value_shapes: list,
        value_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Perform forward pass for multiscale deformable attention.

        Args:
            query (torch.Tensor): Query tensor with shape [bs, query_length, C].
            refer_bbox (torch.Tensor): Reference bounding boxes with shape [bs, query_length, n_levels, 2], range in [0,
                1], top-left (0,0), bottom-right (1, 1), including padding area.
            value (torch.Tensor): Value tensor with shape [bs, value_length, C].
            value_shapes (list): List with shape [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})].
            value_mask (torch.Tensor, optional): Mask tensor with shape [bs, value_length], True for non-padding
                elements, False for padding elements.

        Returns:
            (torch.Tensor): Output tensor with shape [bs, Length_{query}, C].

        References:
            https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
        """
        bs, len_q = query.shape[:2]
        len_v = value.shape[1]
        assert sum(s[0] * s[1] for s in value_shapes) == len_v

        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], float(0))
        value = value.view(bs, len_v, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(bs, len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(bs, len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(bs, len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        num_points = refer_bbox.shape[-1]
        if num_points == 2:
            offset_normalizer = torch.as_tensor(value_shapes, dtype=query.dtype, device=query.device).flip(-1)
            add = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            sampling_locations = refer_bbox[:, :, None, :, None, :] + add
        elif num_points == 4:
            add = sampling_offsets / self.n_points * refer_bbox[:, :, None, :, None, 2:] * 0.5
            sampling_locations = refer_bbox[:, :, None, :, None, :2] + add
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {num_points}.")
        
        # Save sampling locations for debugging (if enabled by parent module)
        if hasattr(self, '_save_sampling_locations') and self._save_sampling_locations:
            self._debug_sampling_locations = sampling_locations.detach().cpu()
            self._debug_sampling_offsets = sampling_offsets.detach().cpu()
            self._debug_attention_weights = attention_weights.detach().cpu()
        
        output = multi_scale_deformable_attn_pytorch(value, value_shapes, sampling_locations, attention_weights)
        return self.output_proj(output)


class DeformableTransformerDecoderLayer(nn.Module):
    """Deformable Transformer Decoder Layer inspired by PaddleDetection and Deformable-DETR implementations.

    This class implements a single decoder layer with self-attention, cross-attention using multiscale deformable
    attention, and a feedforward network.

    Attributes:
        self_attn (nn.MultiheadAttention): Self-attention module.
        dropout1 (nn.Dropout): Dropout after self-attention.
        norm1 (nn.LayerNorm): Layer normalization after self-attention.
        cross_attn (MSDeformAttn): Cross-attention module.
        dropout2 (nn.Dropout): Dropout after cross-attention.
        norm2 (nn.LayerNorm): Layer normalization after cross-attention.
        linear1 (nn.Linear): First linear layer in the feedforward network.
        act (nn.Module): Activation function.
        dropout3 (nn.Dropout): Dropout in the feedforward network.
        linear2 (nn.Linear): Second linear layer in the feedforward network.
        dropout4 (nn.Dropout): Dropout after the feedforward network.
        norm3 (nn.LayerNorm): Layer normalization after the feedforward network.

    References:
        https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
        https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_transformer.py
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ffn: int = 1024,
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        n_levels: int = 4,
        n_points: int = 4,
    ):
        """Initialize the DeformableTransformerDecoderLayer with the given parameters.

        Args:
            d_model (int): Model dimension.
            n_heads (int): Number of attention heads.
            d_ffn (int): Dimension of the feedforward network.
            dropout (float): Dropout probability.
            act (nn.Module): Activation function.
            n_levels (int): Number of feature levels.
            n_points (int): Number of sampling points.
        """
        super().__init__()

        # Self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.act = act
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos: torch.Tensor | None) -> torch.Tensor:
        """Add positional embeddings to the input tensor, if provided."""
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt: torch.Tensor) -> torch.Tensor:
        """Perform forward pass through the Feed-Forward Network part of the layer.

        Args:
            tgt (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after FFN.
        """
        tgt2 = self.linear2(self.dropout3(self.act(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        return self.norm3(tgt)

    def forward(
        self,
        embed: torch.Tensor,
        refer_bbox: torch.Tensor,
        feats: torch.Tensor,
        shapes: list,
        padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Perform the forward pass through the entire decoder layer.

        Args:
            embed (torch.Tensor): Input embeddings.
            refer_bbox (torch.Tensor): Reference bounding boxes.
            feats (torch.Tensor): Feature maps.
            shapes (list): Feature shapes.
            padding_mask (torch.Tensor, optional): Padding mask.
            attn_mask (torch.Tensor, optional): Attention mask.
            query_pos (torch.Tensor, optional): Query position embeddings.

        Returns:
            (torch.Tensor): Output tensor after decoder layer.
        """
        # Self attention
        q = k = self.with_pos_embed(embed, query_pos)
        tgt = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), embed.transpose(0, 1), attn_mask=attn_mask)[
            0
        ].transpose(0, 1)
        embed = embed + self.dropout1(tgt)
        embed = self.norm1(embed)

        # Cross attention
        tgt = self.cross_attn(
            self.with_pos_embed(embed, query_pos), refer_bbox.unsqueeze(2), feats, shapes, padding_mask
        )
        embed = embed + self.dropout2(tgt)
        embed = self.norm2(embed)

        # FFN
        return self.forward_ffn(embed)


class DeformableTransformerDecoder(nn.Module):
    """Deformable Transformer Decoder based on PaddleDetection implementation.

    This class implements a complete deformable transformer decoder with multiple decoder layers and prediction heads
    for bounding box regression and classification.

    Attributes:
        layers (nn.ModuleList): List of decoder layers.
        num_layers (int): Number of decoder layers.
        hidden_dim (int): Hidden dimension.
        eval_idx (int): Index of the layer to use during evaluation.

    References:
        https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    """

    def __init__(self, hidden_dim: int, decoder_layer: nn.Module, num_layers: int, eval_idx: int = -1):
        """Initialize the DeformableTransformerDecoder with the given parameters.

        Args:
            hidden_dim (int): Hidden dimension.
            decoder_layer (nn.Module): Decoder layer module.
            num_layers (int): Number of decoder layers.
            eval_idx (int): Index of the layer to use during evaluation.
        """
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(
        self,
        embed: torch.Tensor,  # decoder embeddings
        refer_bbox: torch.Tensor,  # anchor
        feats: torch.Tensor,  # image features
        shapes: list,  # feature shapes
        bbox_head: nn.Module,
        score_head: nn.Module,
        pos_mlp: nn.Module,
        attn_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ):
        """Perform the forward pass through the entire decoder.

        Args:
            embed (torch.Tensor): Decoder embeddings.
            refer_bbox (torch.Tensor): Reference bounding boxes.
            feats (torch.Tensor): Image features.
            shapes (list): Feature shapes.
            bbox_head (nn.Module): Bounding box prediction head.
            score_head (nn.Module): Score prediction head.
            pos_mlp (nn.Module): Position MLP.
            attn_mask (torch.Tensor, optional): Attention mask.
            padding_mask (torch.Tensor, optional): Padding mask.

        Returns:
            dec_bboxes (torch.Tensor): Decoded bounding boxes.
            dec_cls (torch.Tensor): Decoded classification scores.
        """
        output = embed
        dec_bboxes = []
        dec_cls = []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()
        for i, layer in enumerate(self.layers):
            output = layer(output, refer_bbox, feats, shapes, padding_mask, attn_mask, pos_mlp(refer_bbox))

            bbox = bbox_head[i](output)
            refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox))

            if self.training:
                dec_cls.append(score_head[i](output))
                if i == 0:
                    dec_bboxes.append(refined_bbox)
                else:
                    dec_bboxes.append(torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox)))
            elif i == self.eval_idx:
                dec_cls.append(score_head[i](output))
                dec_bboxes.append(refined_bbox)
                break

            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox

        return torch.stack(dec_bboxes), torch.stack(dec_cls)

    

class MSDeformFusionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        d_model: int = 256,
        num_heads: int = 8,
        d_ffn: int = 1024,
        dropout: float = 0.1,
        n_points: int = 6,
    ):
        super().__init__()
        self.channels = channels
        self.d_model = d_model
        self.n_levels = 2

        self.proj_vis = nn.Linear(channels, d_model)
        self.proj_aux = nn.Linear(channels, d_model)
        self.out_proj  = nn.Linear(d_model, channels)

        self.modal_weight = nn.Parameter(torch.tensor(0.5))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Linear(d_ffn, d_model),
        )

        self.attn = MSDeformAttn(
            d_model, n_levels=self.n_levels, n_heads=num_heads, n_points=n_points
        )

    @staticmethod
    def _to_sequence(x: torch.Tensor) -> torch.Tensor:
        return x.flatten(2).transpose(1, 2)

    @staticmethod
    def _to_featuremap(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        return x.transpose(1, 2).view(x.shape[0], x.shape[2], h, w)

    def _reference_points(self, h: int, w: int, device, dtype) -> torch.Tensor:
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0.5 / h, 1.0 - 0.5 / h, h, device=device, dtype=dtype),
            torch.linspace(0.5 / w, 1.0 - 0.5 / w, w, device=device, dtype=dtype),
            indexing="ij",
        )
        coords = torch.stack((grid_x, grid_y), dim=-1)
        return coords.reshape(1, h * w, 1, 2).expand(1, -1, self.n_levels, -1)

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        if len(x) == 1:
            return x[0]

        feat_vis, feat_aux = x
        B, _, H, W = feat_vis.shape

        seq_vis = self._to_sequence(feat_vis)
        seq_aux = self._to_sequence(feat_aux)

        vis_proj = self.proj_vis(seq_vis)
        aux_proj = self.proj_aux(seq_aux)

        w = torch.sigmoid(self.modal_weight)
        raw_query = w * vis_proj + (1.0 - w) * aux_proj

        # value must match d_model for MSDeformAttn.value_proj
        value = torch.cat([vis_proj, aux_proj], dim=1)
        value_shapes = [(H, W), (H, W)]

        ref_pts = self._reference_points(H, W, raw_query.device, raw_query.dtype)
        ref_pts = ref_pts.expand(B, -1, -1, -1)

        attn_out = self.attn(self.norm1(raw_query), ref_pts, value, value_shapes)
        fused = raw_query + self.dropout(attn_out)

        fused = fused + self.dropout(self.ffn(self.norm2(fused)))

        out = self.out_proj(fused)
        return self._to_featuremap(out, H, W)

class Wavelet2D(nn.Module):
    """General 2D Discrete Wavelet Transform supporting multiple wavelet families.
    
    Decomposes input into LL (low-frequency), LH (horizontal edges),
    HL (vertical edges), and HH (diagonal edges) sub-bands using separable
    2D filters constructed from 1D wavelet filter banks.
    
    Supported wavelets:
        - 'haar': Haar wavelet (2-tap, same as db1)
        - 'db2': Daubechies-2 (4-tap)
        - 'db3': Daubechies-3 (6-tap)
        - 'db4': Daubechies-4 (8-tap)
        - 'sym4': Symlet-4 (8-tap)
        - 'bior1.3': Biorthogonal-1.3 (6-tap)
        - 'bior2.2': Biorthogonal-2.2 (6-tap)
        - 'coif1': Coiflet-1 (6-tap)
    
    Args:
        wavelet (str): Wavelet family name. Default: 'db2'.
    """
    
    # Decomposition filter coefficients for each wavelet family
    WAVELET_FILTERS = {
        'haar': {
            'lo': [0.7071067811865476, 0.7071067811865476],
            'hi': [-0.7071067811865476, 0.7071067811865476],
        },
        'db2': {
            'lo': [-0.12940952255126037, 0.2241438680420134,
                   0.8365163037378079, 0.48296291314453416],
            'hi': [-0.48296291314453416, 0.8365163037378079,
                   -0.2241438680420134, -0.12940952255126037],
        },
        'db3': {
            'lo': [0.035226291882100656, -0.08544127388224149,
                   -0.13501102001039084, 0.4598775021193313,
                   0.8068915093133388, 0.3326705529509569],
            'hi': [-0.3326705529509569, 0.8068915093133388,
                   -0.4598775021193313, -0.13501102001039084,
                   0.08544127388224149, 0.035226291882100656],
        },
        'db4': {
            'lo': [-0.010597401785069032, 0.0328830116668852, 0.030841381835560764, -0.18703481171909309,
                   -0.027983769416859854, 0.6308807679298589, 0.7148465705529157, 0.2303778133088965],
            'hi': [-0.2303778133088965, 0.7148465705529157, -0.6308807679298589, -0.027983769416859854,
                   0.18703481171909309, 0.030841381835560764, -0.0328830116668852, -0.010597401785069032],
        },
        'sym4': {
            'lo': [-0.07576571478927333, -0.02963552764599851, 0.49761866763201545, 0.8037387518059161,
                   0.29785779560527736, -0.09921954357684722, -0.012603967262037833, 0.0322231006040427],
            'hi': [-0.0322231006040427, -0.012603967262037833, 0.09921954357684722, 0.29785779560527736,
                   -0.8037387518059161, 0.49761866763201545, 0.02963552764599851, -0.07576571478927333],
        },
        'bior1.3': {
            'lo': [-0.08838834764831845, 0.08838834764831845, 0.7071067811865476, 0.7071067811865476,
                   0.08838834764831845, -0.08838834764831845],
            'hi': [0.0, 0.0, -0.7071067811865476, 0.7071067811865476, 0.0, 0.0],
        },
        'bior2.2': {
            'lo': [0.0, -0.1767766952966369, 0.3535533905932738, 1.0606601717798212,
                   0.3535533905932738, -0.1767766952966369],
            'hi': [0.0, 0.3535533905932738, -0.7071067811865476, 0.3535533905932738, 0.0, 0.0],
        },
        'coif1': {
            'lo': [-0.01565572813546454, -0.07273261951285131,
                   0.38486484686420286, 0.8525720202122554,
                   0.33789766245780922, -0.07273261951285131],
            'hi': [0.07273261951285131, 0.33789766245780922,
                   -0.8525720202122554, 0.38486484686420286,
                   0.07273261951285131, -0.01565572813546454],
        },
    }
    
    def __init__(self, wavelet: str = 'db2'):
        super().__init__()
        if wavelet not in self.WAVELET_FILTERS:
            raise ValueError(
                f"Unsupported wavelet '{wavelet}'. "
                f"Choose from: {list(self.WAVELET_FILTERS.keys())}"
            )
        
        self.wavelet_name = wavelet
        filters = self.WAVELET_FILTERS[wavelet]
        lo = torch.tensor(filters['lo'], dtype=torch.float32)
        hi = torch.tensor(filters['hi'], dtype=torch.float32)
        
        # Build 2D filters via outer product of 1D filters
        # LL: low-pass both directions, LH: high-col low-row,
        # HL: low-col high-row, HH: high-pass both
        self.register_buffer('ll_filter', torch.outer(lo, lo))
        self.register_buffer('lh_filter', torch.outer(hi, lo))
        self.register_buffer('hl_filter', torch.outer(lo, hi))
        self.register_buffer('hh_filter', torch.outer(hi, hi))
        
        self.filter_size = len(lo)
        self.pad_size = (self.filter_size - 2) // 2
    
    def forward(self, x: torch.Tensor) -> tuple:
        """Perform 2D wavelet decomposition.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            
        Returns:
            tuple: (LL, LH, HL, HH) each of shape (B, C, H//2, W//2).
        """
        B, C, H, W = x.shape
        
        # Pad if dimensions are odd
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        
        # Reflect pad for filter overlap (needed for filters longer than 2)
        if self.pad_size > 0:
            x = F.pad(x, (self.pad_size, self.pad_size,
                          self.pad_size, self.pad_size), mode='reflect')
        
        # Reshape filters for depthwise conv: (C, 1, K, K)
        K = self.filter_size
        ll_filter = self.ll_filter.to(dtype=x.dtype, device=x.device)
        lh_filter = self.lh_filter.to(dtype=x.dtype, device=x.device)
        hl_filter = self.hl_filter.to(dtype=x.dtype, device=x.device)
        hh_filter = self.hh_filter.to(dtype=x.dtype, device=x.device)
        ll = ll_filter.unsqueeze(0).unsqueeze(0).expand(C, -1, -1, -1)
        lh = lh_filter.unsqueeze(0).unsqueeze(0).expand(C, -1, -1, -1)
        hl = hl_filter.unsqueeze(0).unsqueeze(0).expand(C, -1, -1, -1)
        hh = hh_filter.unsqueeze(0).unsqueeze(0).expand(C, -1, -1, -1)
        
        # Apply filters with stride 2 (downsampling)
        LL = F.conv2d(x, ll, stride=2, groups=C)
        LH = F.conv2d(x, lh, stride=2, groups=C)
        HL = F.conv2d(x, hl, stride=2, groups=C)
        HH = F.conv2d(x, hh, stride=2, groups=C)
        
        return LL, LH, HL, HH


class WaveletOffsetMSDeformAttn(nn.Module):
    """MSDeformAttn variant whose ``sampling_offsets`` are predicted from wavelet HF/LF energy.

    Structurally identical to :class:`MSDeformAttn`. The only change is the input that drives
    the offset Linear: instead of one ``nn.Linear(d_model, n_heads*n_levels*n_points*2)`` fed by
    the query, we use two per-level ``nn.Linear`` layers fed by per-modality frequency energy:
    - level 0 (VIS) : visible low-frequency energy (LL)
    - level 1 (IR)  : infrared high-frequency energy (LH + HL + HH)

    Everything else (value_proj, attention_weights, output_proj, radial offset init,
    0.1 gradient scaling) mirrors the original MSDeformAttn exactly.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_levels: int = 2,
        n_heads: int = 8,
        n_points: int = 4,
        wavelet_type: str = "haar",
        wavelet_channels: int | None = None,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, but got {d_model} and {n_heads}")
        _d_per_head = d_model // n_heads
        assert _d_per_head * n_heads == d_model, "`d_model` must be divisible by `n_heads`"
        if n_levels != 2:
            raise ValueError(f"WaveletOffsetMSDeformAttn expects n_levels=2 (VIS/IR), but got {n_levels}.")

        self.im2col_step = 64
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.wavelet_channels = d_model if wavelet_channels is None else wavelet_channels

        # Non-learnable wavelet decomposer used to build energy tokens.
        self.wavelet = Wavelet2D(wavelet=wavelet_type)

        # === Replaced layer (vs MSDeformAttn) ============================
        # Original: self.sampling_offsets = nn.Linear(d_model, n_heads*n_levels*n_points*2)
        # New: per-level Linear, input = per-modality wavelet energy token.
        self.sampling_offsets_vis = nn.Linear(self.wavelet_channels, n_heads * n_points * 2)
        self.sampling_offsets_ir = nn.Linear(self.wavelet_channels, n_heads * n_points * 2)
        # =================================================================
        
        # Normalize wavelet energy tokens to stabilize offset prediction
        self.wavelet_norm_vis = nn.LayerNorm(self.wavelet_channels)
        self.wavelet_norm_ir = nn.LayerNorm(self.wavelet_channels)

        self.ir_lca_dilation = 3
        self.ir_lca_eps = 1e-6
        self.vis_lowpass_radius = 1
        self.vis_guided_eps = 1e-3
        self.ir_contrast_enhance = nn.Parameter(torch.tensor(0.1))
        # Unchanged from MSDeformAttn:
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        """Mirror MSDeformAttn's offset bias init, split across the two per-level Linears."""
        constant_(self.sampling_offsets_vis.weight.data, 0.0)
        constant_(self.sampling_offsets_ir.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets_vis.bias = nn.Parameter(grid_init[:, 0].contiguous().view(-1))
            self.sampling_offsets_ir.bias = nn.Parameter(grid_init[:, 1].contiguous().view(-1))

        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

        # Same 0.1 gradient scaling as MSDeformAttn
        self.sampling_offsets_vis.weight.register_hook(lambda g: g * 0.1)
        self.sampling_offsets_vis.bias.register_hook(lambda g: g * 0.1)
        self.sampling_offsets_ir.weight.register_hook(lambda g: g * 0.1)
        self.sampling_offsets_ir.bias.register_hook(lambda g: g * 0.1)

    @staticmethod
    def _normalize_energy(energy: torch.Tensor) -> torch.Tensor:
        """Normalize each channel map so offsets use relative structural saliency."""
        mean = energy.mean(dim=(-2, -1), keepdim=True)
        std = energy.std(dim=(-2, -1), keepdim=True, unbiased=False)
        return (energy - mean) / (std + 1e-6)

    @staticmethod
    def _shift_feature(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
        """Shift feature map with replicated borders for local directional contrast."""
        B, C, H, W = x.shape
        pad_l = max(dx, 0)
        pad_r = max(-dx, 0)
        pad_t = max(dy, 0)
        pad_b = max(-dy, 0)
        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode="replicate")
        y_start = pad_b
        x_start = pad_r
        return x[:, :, y_start:y_start + H, x_start:x_start + W]

    def _enhance_ir_high_energy(self, ir_mag: torch.Tensor) -> torch.Tensor:
        """LCA-guided enhancement for weak, low-contrast IR high-frequency magnitude."""
        out_dtype = ir_mag.dtype
        ir_mag = ir_mag.float()
        base = ir_mag
        d = self.ir_lca_dilation
        nb_1 = 0.5 * (self._shift_feature(ir_mag, -d, -d) + self._shift_feature(ir_mag, d, d))
        nb_2 = 0.5 * (self._shift_feature(ir_mag, -d, d) + self._shift_feature(ir_mag, d, -d))
        nb_3 = 0.5 * (self._shift_feature(ir_mag, 0, -d) + self._shift_feature(ir_mag, 0, d))
        nb_4 = 0.5 * (self._shift_feature(ir_mag, -d, 0) + self._shift_feature(ir_mag, d, 0))

        d1 = torch.relu(ir_mag - nb_1)
        d2 = torch.relu(ir_mag - nb_2)
        d3 = torch.relu(ir_mag - nb_3)
        d4 = torch.relu(ir_mag - nb_4)
        score = d1 * d2 + d3 * d4

        score_mean = score.mean(dim=(-2, -1), keepdim=True)
        score_std = score.std(dim=(-2, -1), keepdim=True, unbiased=False)
        score = (score - score_mean) / (score_std + self.ir_lca_eps)
        gate = torch.relu(2.0 * torch.sigmoid(score) - 1.0)  #只增强高于平均的 LCA 响应

        return torch.log1p(base * gate).to(out_dtype)

    @staticmethod
    def _box_mean(x: torch.Tensor, radius: int) -> torch.Tensor:
        """Reflect-padded local mean used by the guided low-pass filter."""
        k = 2 * radius + 1
        x = F.pad(x, (radius, radius, radius, radius), mode="reflect")
        return F.avg_pool2d(x, kernel_size=k, stride=1, padding=0)

    def _lowpass_visible_feature(self, vis_feat: torch.Tensor, radius: int) -> torch.Tensor:
        """Guided-filter low-pass base estimation for visible features only."""
        guide = vis_feat
        src = vis_feat
        mean_g = self._box_mean(guide, radius)
        mean_s = self._box_mean(src, radius)
        var_g = self._box_mean(guide * guide, radius) - mean_g.pow(2)
        cov_gs = self._box_mean(guide * src, radius) - mean_g * mean_s
        a = cov_gs / (var_g + self.vis_guided_eps)
        b = mean_s - a * mean_g
        mean_a = self._box_mean(a, radius)
        mean_b = self._box_mean(b, radius)
        return mean_a * guide + mean_b

    def _wavelet_tokens(self, vis_feat: torch.Tensor, aux_feat: torch.Tensor):
        """Build robust VIS-low and IR-high frequency energy tokens at original resolution."""
        B, C, H, W = vis_feat.shape

        vis_base = self._lowpass_visible_feature(vis_feat, self.vis_lowpass_radius)
        vis_feat = vis_feat - vis_base
        v_LL, v_LH, v_HL, v_HH = self.wavelet(vis_feat)
        i_LL, i_LH, i_HL, i_HH = self.wavelet(aux_feat)

        vis_low = v_LL.pow(2)

        ir_high = i_LH.pow(2) + i_HL.pow(2) + i_HH.pow(2)

        lh_residual = self._enhance_ir_high_energy(i_LH.abs())
        hl_residual = self._enhance_ir_high_energy(i_HL.abs())
        hh_residual = self._enhance_ir_high_energy(i_HH.abs())
        ir_residual = lh_residual + hl_residual + hh_residual
        ir_high = ir_high + torch.tanh(self.ir_contrast_enhance) * ir_residual

        vis_low = F.interpolate(vis_low, size=(H, W), mode="bilinear", align_corners=False)
        ir_high = F.interpolate(ir_high, size=(H, W), mode="bilinear", align_corners=False)
        
        vis_token = vis_low.flatten(2).transpose(1, 2).contiguous()        # [B, H*W, C]
        ir_token = ir_high.flatten(2).transpose(1, 2).contiguous()         # [B, H*W, C]
        vis_token = self.wavelet_norm_vis(vis_token)
        ir_token = self.wavelet_norm_ir(ir_token)
        return vis_token, ir_token

    def forward(
        self,
        query: torch.Tensor,
        refer_bbox: torch.Tensor,
        value: torch.Tensor,
        value_shapes: list,
        wavelet_feats: tuple[torch.Tensor, torch.Tensor],
        value_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Same as MSDeformAttn.forward, except sampling_offsets are wavelet-driven."""
        bs, len_q = query.shape[:2]
        len_v = value.shape[1]
        assert sum(s[0] * s[1] for s in value_shapes) == len_v

        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], float(0))
        value = value.view(bs, len_v, self.n_heads, self.d_model // self.n_heads)

        # === Replaced part: sampling_offsets from wavelet energy =========
        vis_token, ir_token = self._wavelet_tokens(*wavelet_feats)
        vis_token = vis_token.to(dtype=query.dtype, device=query.device)
        ir_token = ir_token.to(dtype=query.dtype, device=query.device)
        if vis_token.shape[1] != len_q or ir_token.shape[1] != len_q:
            raise ValueError(
                f"Wavelet token lengths {(vis_token.shape[1], ir_token.shape[1])} "
                f"must match query length {len_q}."
            )
        vis_offsets = self.sampling_offsets_vis(vis_token).view(bs, len_q, self.n_heads, self.n_points, 2)
        ir_offsets = self.sampling_offsets_ir(ir_token).view(bs, len_q, self.n_heads, self.n_points, 2)
        sampling_offsets = torch.stack([vis_offsets, ir_offsets], dim=3)
        # shape: [bs, len_q, n_heads, n_levels=2, n_points, 2]
        # =================================================================

        attention_weights = self.attention_weights(query).view(bs, len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(bs, len_q, self.n_heads, self.n_levels, self.n_points)

        num_points = refer_bbox.shape[-1]
        if num_points == 2:
            offset_normalizer = torch.as_tensor(value_shapes, dtype=query.dtype, device=query.device).flip(-1)
            add = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            sampling_locations = refer_bbox[:, :, None, :, None, :] + add
        elif num_points == 4:
            add = sampling_offsets / self.n_points * refer_bbox[:, :, None, :, None, 2:] * 0.5
            sampling_locations = refer_bbox[:, :, None, :, None, :2] + add
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {num_points}.")

        if hasattr(self, "_save_sampling_locations") and self._save_sampling_locations:
            self._debug_sampling_locations = sampling_locations.detach().cpu()
            self._debug_sampling_offsets = sampling_offsets.detach().cpu()
            self._debug_attention_weights = attention_weights.detach().cpu()

        output = multi_scale_deformable_attn_pytorch(value, value_shapes, sampling_locations, attention_weights)
        return self.output_proj(output)


class WaveletGuidedMSDeformFusionBlock(nn.Module):
    """
    Args:
        channels (int): Input channel dimension for the modalities.
        d_model (int): Transformer hidden size.
        num_heads (int): Attention heads for MSDeformAttn.
        d_ffn (int): Hidden size of the feed-forward network.
        dropout (float): Dropout applied after attention/FFN.
        n_points (int): Sampling points per head/level for MSDeformAttn.
        wavelet_type (str): Wavelet family (haar, db2, db3, sym3, coif1).
    """
    
    def __init__(
        self,
        channels: int,
        d_model: int = 256,
        num_heads: int = 8,
        d_ffn: int = 1024,
        dropout: float = 0.1,
        n_points: int = 6,
        wavelet_type: str = 'haar',
    ):
        super().__init__()
        self.channels = channels
        self.d_model = d_model
        self.n_levels = 2  # visual + IR modality
        
        # Input projections
        self.proj_vis = nn.Linear(channels, d_model)
        self.proj_aux = nn.Linear(channels, d_model)
        self.out_proj = nn.Linear(d_model, channels)

        # Learnable modal weight (same as MSDeformFusionBlock)
        self.modal_weight = nn.Parameter(torch.tensor(0.5))

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Use wavelet frequency tokens to generate deformable sampling offsets.
        self.deform_attn = WaveletOffsetMSDeformAttn(
            d_model=d_model,
            n_levels=self.n_levels,
            n_heads=num_heads,
            n_points=n_points,
            wavelet_type=wavelet_type,
            wavelet_channels=channels,
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Linear(d_ffn, d_model),
        )
    
    @staticmethod
    def _to_sequence(x: torch.Tensor) -> torch.Tensor:
        """Convert feature map to sequence: [B, C, H, W] -> [B, H*W, C]."""
        B, C, H, W = x.shape
        return x.flatten(2).transpose(1, 2)
    
    @staticmethod
    def _to_featuremap(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Convert sequence to feature map: [B, H*W, C] -> [B, C, H, W]."""
        B, L, C = x.shape
        return x.transpose(1, 2).view(B, C, h, w)
    
    def _reference_points(self, h: int, w: int, device, dtype) -> torch.Tensor:
        """Generate reference points grid for deformable attention."""
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0.5 / h, 1.0 - 0.5 / h, h, device=device, dtype=dtype),
            torch.linspace(0.5 / w, 1.0 - 0.5 / w, w, device=device, dtype=dtype),
            indexing="ij",
        )
        coords = torch.stack((grid_x, grid_y), dim=-1)
        coords = coords.view(1, h * w, 1, 2)
        return coords.repeat(1, 1, self.n_levels, 1)
    
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass for wavelet-guided multimodal fusion.
        
        Args:
            x (list): List of two tensors [feat_vis, feat_aux], each [B, C, H, W]
            
        Returns:
            torch.Tensor: Fused feature map [B, C, H, W]
        """
        if len(x) == 1:
            return x[0]
        
        feat_vis, feat_aux = x
        B, _, H, W = feat_vis.shape

        # Convert to sequences
        seq_vis = self.proj_vis(self._to_sequence(feat_vis))
        seq_aux = self.proj_aux(self._to_sequence(feat_aux))
        
        # ===== UNIFIED QUERY FEATURE (learnable modal weight) =====
        w = torch.sigmoid(self.modal_weight)
        seq_query = w * seq_vis + (1.0 - w) * seq_aux
        seq_query_norm = self.norm1(seq_query)

        # Concatenate values from both modalities
        value = torch.cat([seq_vis, seq_aux], dim=1)
        value_shapes = [(H, W)] * self.n_levels

        reference_points = self._reference_points(H, W, seq_vis.device, seq_vis.dtype)
        reference_points = reference_points.expand(B, -1, -1, -1)

        # Use wavelet-frequency-guided sampling offsets instead of query-linear offsets.
        attn_out = self.deform_attn(
            seq_query_norm,
            reference_points,
            value,
            value_shapes,
            wavelet_feats=(feat_vis, feat_aux),
        )
        
        fused = seq_query + self.dropout(attn_out)

        fused = fused + self.dropout(self.ffn(self.norm2(fused)))

        # Convert back to feature map
        out = self.out_proj(fused)
        return self._to_featuremap(out, H, W)
