# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type, Optional

from segment_anything.modeling.common import LayerNorm2d

class GaussianSmoothing(nn.Module):
    """Applies depthwise Gaussian smoothing (fixed kernel) to logits in a differentiable way."""
    def __init__(self, channels: int, kernel_size: int = 5, sigma: float = 1.0):
        super().__init__()
        # 1D Gaussian
        half = (kernel_size - 1) / 2.
        x = torch.arange(-half, half + 1)
        gauss = torch.exp(-x.pow(2) / (2 * sigma**2))
        gauss = gauss / gauss.sum()
        # separable 2D kernel
        kernel2d = gauss[:, None] @ gauss[None, :]
        kernel2d = kernel2d.unsqueeze(0).unsqueeze(0)  # (1,1,K,K)
        kernel2d = kernel2d.repeat(channels, 1, 1, 1)  # (channels,1,K,K)
        self.register_buffer('weight', kernel2d)
        self.groups = channels
        self.padding = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, channels, H, W) logits
        return F.conv2d(x, self.weight, bias=None,
                         stride=1, padding=self.padding,
                         groups=self.groups)

class MaskDecoder3(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        num_classes: int = 4,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        tranformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer


        self.num_multimask_outputs = num_multimask_outputs
        self.num_classes = num_classes

        # Gabor fusion: project 12-channel Gabor maps into transformer_dim
        self.texture_proj = nn.Sequential(
            nn.Conv2d(12, transformer_dim, kernel_size=1, bias=False),
            nn.GELU(),                # ← add activation here
        )

        # — gating module: takes [img; tex] → per‐channel gate in [0,1] —
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(transformer_dim * 2, transformer_dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # mask tokens
        self.iou_token = nn.Embedding(num_classes, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(num_classes*self.num_mask_tokens, transformer_dim)

        # upscaling head
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        gabor_feats: Optional[torch.Tensor] = None,
        multimask_output: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks_1(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            gabor_feats=gabor_feats,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred



    def predict_masks_1(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        gabor_feats: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""


        # Concatenate output tokens
        mask_tokens = self.mask_tokens.weight.view(-1, self.num_mask_tokens, self.transformer_dim)
        output_tokens = torch.cat([self.iou_token.weight.unsqueeze(1), mask_tokens], dim=1)
        tokens = output_tokens.repeat(image_embeddings.size(0), 1, 1)

        # expand image embeddings per mask

        # Expand per-image data in batch direction to be per-mask
        # src = image_embeddings.expand(-1, output_tokens.shape[0], -1, -1, -1).flatten(0, 1)
        # pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)


        # Expand per-image data in batch direction to be per-mask
        # src = image_embeddings.expand(-1, output_tokens.shape[0], -1, -1, -1)
        # Bm, M, D, Hf, Wf = src.shape
        # src = src.flatten(0, 1)  # (B*M, D, H', W')

        img_emb = image_embeddings.squeeze(1)            # (B, D, Hf, Wf)

        # fuse Gabor if provided
        if gabor_feats is not None:
            # project and expand
            tex_emb = self.texture_proj(gabor_feats)              # (B, D, H', W')

            # 2) build gate
            comb = torch.cat([img_emb, tex_emb], dim=1)   # (B, 2D, Hf, Wf)
            gate = self.fusion_gate(comb)                # (B, D, Hf, Wf), in (0,1)


            # tex = tex.unsqueeze(1).expand(-1, M, -1, -1, -1)  # (B, M, D, H', W')
            # tex = tex.flatten(0, 1)                           # (B*M, D, H', W')
            # src = src + tex

            # 3) gated fusion
            fused = gate * img_emb + (1 - gate) * tex_emb
        else:
            fused = img_emb
        
        fused = fused.unsqueeze(1).expand(-1, output_tokens.shape[0], -1, -1, -1).flatten(0, 1)  # (B*M, D, H', W')

        
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)

        b, c, h, w = fused.shape

        # Run the transformer
        hs, src = self.transformer(fused, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred

class MaskDecoder2(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        num_classes: int = 4,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        tranformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer


        self.num_multimask_outputs = num_multimask_outputs
        self.num_classes = num_classes

        # Gabor fusion: project 12-channel Gabor maps into transformer_dim
        self.texture_proj = nn.Sequential(
            nn.Conv2d(12, transformer_dim, kernel_size=1, bias=False),
            nn.GELU(),                # ← add activation here
        )

        # mask tokens
        self.iou_token = nn.Embedding(num_classes, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(num_classes*self.num_mask_tokens, transformer_dim)

        # upscaling head
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        gabor_feats: Optional[torch.Tensor] = None,
        multimask_output: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks_1(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            gabor_feats=gabor_feats,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred



    def predict_masks_1(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        gabor_feats: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""


        # Concatenate output tokens
        mask_tokens = self.mask_tokens.weight.view(-1, self.num_mask_tokens, self.transformer_dim)
        output_tokens = torch.cat([self.iou_token.weight.unsqueeze(1), mask_tokens], dim=1)
        tokens = output_tokens.repeat(image_embeddings.size(0), 1, 1)

        # expand image embeddings per mask

        # Expand per-image data in batch direction to be per-mask
        # src = image_embeddings.expand(-1, output_tokens.shape[0], -1, -1, -1).flatten(0, 1)
        # pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)


        # Expand per-image data in batch direction to be per-mask
        src = image_embeddings.expand(-1, output_tokens.shape[0], -1, -1, -1)
        Bm, M, D, Hf, Wf = src.shape
        src = src.flatten(0, 1)  # (B*M, D, H', W')

        # fuse Gabor if provided
        if gabor_feats is not None:
            # project and expand
            tex = self.texture_proj(gabor_feats)              # (B, D, H', W')
            tex = tex.unsqueeze(1).expand(-1, M, -1, -1, -1)  # (B, M, D, H', W')
            tex = tex.flatten(0, 1)                           # (B*M, D, H', W')
            src = src + tex

        
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)

        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred
    

class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        num_classes: int = 4,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        tranformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs
        self.num_classes = num_classes

        self.iou_token = nn.Embedding(num_classes, transformer_dim)
        #self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(num_classes*self.num_mask_tokens, transformer_dim)
        #self.mask_tokens = nn.Embedding(num_classes, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        multimask_output: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks_1(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
        )

        # Select the correct mask or masks for outptu
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    

    def predict_masks_1(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        mask_tokens = self.mask_tokens.weight.view(-1, self.num_mask_tokens, self.transformer_dim)
        output_tokens = torch.cat([self.iou_token.weight.unsqueeze(1), mask_tokens], dim=1)
        tokens = output_tokens.repeat(image_embeddings.size(0), 1, 1)

        # Expand per-image data in batch direction to be per-mask
        src = image_embeddings.expand(-1, output_tokens.shape[0], -1, -1, -1).flatten(0, 1)
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)

        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred
    

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens

        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0).unsqueeze(0)
        tokens = output_tokens.repeat(image_embeddings.size(0), 1, 1)

        # Expand per-image data in batch direction to be per-mask
        src = image_embeddings.squeeze(1)
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)

        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_classes), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_classes):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)

        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred

# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
