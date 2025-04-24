# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type, Optional

from segment_anything.modeling.common import LayerNorm2d


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
        texture_num_heads: int = 8, 
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

        # 1) texture projection
        self.texture_proj = nn.Sequential(
            nn.Conv2d(12, transformer_dim, kernel_size=1, bias=False),
            nn.GELU(),
        )
        # 2) cross-attention module
        self.texture_attention = nn.MultiheadAttention(
            embed_dim=transformer_dim,
            num_heads=texture_num_heads,
            batch_first=False   # because we'll supply seq_len×batch×embed
        )


        self.num_multimask_outputs = num_multimask_outputs
        self.num_classes = num_classes


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

        B, _, D, Hf, Wf = image_embeddings.shape
        # 1) cross-attn fusion as before …
        if gabor_feats is not None:
            tex = self.texture_proj(gabor_feats)                          # (B, D, Hf, Wf)
            T = Hf * Wf
            tex_seq = tex.view(B, D, T).permute(2,0,1)                     # (T, B, D)
            img_seq = image_embeddings.squeeze(1).view(B, D, T).permute(2,0,1)
            attn_out, _ = self.texture_attention( query=img_seq,
                                                  key=tex_seq,
                                                  value=tex_seq )
            attn_out = attn_out.permute(1,2,0).view(B, D, Hf, Wf)
            fused_emb = image_embeddings.squeeze(1) + attn_out
        else:
            fused_emb = image_embeddings.squeeze(1)

        ## tokens: batch × tokens × dim
        all_tokens = torch.cat([
            self.iou_token.weight,       # (num_classes,    D)
            self.mask_tokens.weight      # (num_classes*num_mask_tokens, D)
        ], dim=0)                        # → (M, D)
        tokens = all_tokens.unsqueeze(0).repeat(B, 1, 1)  # (B, M, D)

        # 3) expand fused embeddings to match tokens
        src = fused_emb.unsqueeze(1).expand(-1, tokens.shape[0] // B, -1, -1, -1)
        src = src.flatten(0, 1)                           # (B*M, D, Hf, Wf)

        # 4) positional embeddings
        # flatten your image_pe (1,D,Hf,Wf) → (T,1,D) then repeat B*M/T times
        T = Hf * Wf
        pe_seq = image_pe.view(1, D, T).permute(2,0,1)     # (T, 1, D)
        pe_seq = pe_seq.repeat(1, tokens.shape[0], 1)      # (T, B*M, D)


        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pe_seq, tokens)
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
        B, _, D, Hf, Wf = image_embeddings.shape

        # 1) Cross‐attention fusion (unchanged)
        if gabor_feats is not None:
            tex = self.texture_proj(gabor_feats)                       # (B, D, Hf, Wf)
            T = Hf * Wf
            tex_seq = tex.view(B, D, T).permute(2,0,1)                  # (T, B, D)
            img_seq = image_embeddings.squeeze(1).view(B, D, T).permute(2,0,1)
            attn_out, _ = self.texture_attention(q=img_seq, k=tex_seq, v=tex_seq)
            attn_out = attn_out.permute(1,2,0).view(B, D, Hf, Wf)
            fused_emb = image_embeddings.squeeze(1) + attn_out
        else:
            fused_emb = image_embeddings.squeeze(1)                    # (B, D, Hf, Wf)

        # 2) Build tokens exactly as SAM does
        all_tokens = torch.cat([
            self.iou_token.weight,      # (num_classes, D)
            self.mask_tokens.weight     # (num_classes*num_mask_tokens, D)
        ], dim=0)                       # → (M, D)
        tokens = all_tokens.unsqueeze(0).repeat(B, 1, 1)  # (B, M, D)

        # 3) Call transformer with the right shapes—no flatten!
        hs, src_out = self.transformer(
            fused_emb,   # (B, D, Hf, Wf)
            image_pe,    # (1, D, Hf, Wf)
            tokens       # (B, M, D)
        )

        # 4) Extract IoU and mask tokens & upscale as before
        iou_token_out   = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]

        # src_out is back to (B, D, Hf, Wf)
        up = self.output_upscaling(src_out)
        hyper = torch.stack([
            mlp(mask_tokens_out[:, i, :]) for i, mlp in enumerate(self.output_hypernetworks_mlps)
        ], dim=1)  # (B, num_mask_tokens, transformer_dim//8)

        b, c, h, w = up.shape
        masks = (hyper @ up.view(b, c, h*w)).view(b, -1, h, w)
        iou   = self.iou_prediction_head(iou_token_out)

        return masks, iou
    

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
