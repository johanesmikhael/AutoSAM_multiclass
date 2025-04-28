from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from segment_anything.modeling.prompt_encoder import PositionEmbeddingRandom
from segment_anything.modeling.common import LayerNorm2d


def make_gabor_bank(
    ksize: int,
    sigmas: List[float],
    thetas: List[float],
    lambd: float,
    gamma: float = 0.5,
    psi: float = 0.0,
) -> torch.Tensor:
    """
    Create a (F, 1, ksize, ksize) tensor of even-phase Gabor kernels,
    where F = len(sigmas) * len(thetas).
    """
    half = ksize // 2
    xs = np.linspace(-half, half, ksize)
    ys = xs
    y, x = np.meshgrid(ys, xs)               # (ksize, ksize)
    kernels = []
    for sigma in sigmas:
        for theta in thetas:
            # rotate coordinates
            x_t = x * np.cos(theta) + y * np.sin(theta)
            y_t = -x * np.sin(theta) + y * np.cos(theta)
            # Gabor formula (even phase)
            gb = np.exp(-(x_t**2 + (gamma * y_t)**2) / (2 * sigma**2)) \
                 * np.cos(2 * np.pi * x_t / lambd + psi)
            # normalize
            gb -= gb.mean()
            gb /= np.linalg.norm(gb)
            kernels.append(gb)
    bank = np.stack(kernels, axis=0)         # (F, ksize, ksize)
    bank = bank[:, None, :, :]               # (F, 1, ksize, ksize)
    return torch.from_numpy(bank).float()



class AutoSamSegGabor(nn.Module):
    def __init__(
        self,
        image_encoder: nn.Module,
        seg_decoder: nn.Module,
        img_size: int = 1024,
    ):
        super().__init__()
        self.img_size = img_size
        self.image_encoder = image_encoder
        self.mask_decoder = seg_decoder
        self.pe_layer = PositionEmbeddingRandom(128)

        # 4 orientations × 3 scales = 12 filters
        orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        scales       = [8, 16, 32]
        gabor_bank = make_gabor_bank(
            ksize=31,
            sigmas=[0.56 * s for s in scales],
            thetas=orientations,
            lambd=6.0,  #it was 16.0
        )  # → (12,1,31,31)
        self.register_buffer('gabor_kernels', gabor_bank)

        # Small conv: 12->out_chans
        out_chans = self.image_encoder.neck[0].out_channels
        self.tex_conv = nn.Sequential(
            nn.Conv2d(12, out_chans, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
        )


    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        original_size = W

        # 1) resize for encoder
        x = F.interpolate(
            x,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )

        x_reduced = F.interpolate(
            x,
            (256, 256),
            mode="bilinear",
            align_corners=False,
        )


        # 2) frozen image features
        with torch.no_grad():
            img_emb = self.image_encoder(x)  # (B,D,H',W')

            # 3) compute texture map
            gray = (0.2989 * x_reduced[:,0]
                + 0.5870 * x_reduced[:,1]
                + 0.1140 * x_reduced[:,2]).unsqueeze(1)  # (B,1,H',W')
            tex = F.conv2d(
                gray,
                self.gabor_kernels,
                padding=self.gabor_kernels.shape[-1]//2
            )  # (B,12,H',W')
            
        
        Hf, Wf = img_emb.shape[-2:]

        tex_small = F.interpolate(tex, (Hf,Hf), mode='bilinear', align_corners=False)
        tex_emb = self.tex_conv(tex_small)  # (B, out_chans, Hp, Hp)

        # 5) fuse
        fused = img_emb + tex_emb  # (B,D,H',W')

        # 5) positional encoding
        Hf, Wf = fused.shape[-2:]
        img_pe = self.pe_layer([Hf, Wf]).unsqueeze(0)  # (1,D,Hf,Wf)

        # 7) decode
        mask, iou_pred = self.mask_decoder(
            image_embeddings=fused.unsqueeze(1),
            image_pe=img_pe,
        )

        # 8) resize back
        if mask.shape[-1] != original_size:
            mask = F.interpolate(
                mask, (original_size,)*2,
                mode='bilinear', align_corners=False
            )
        return mask, iou_pred

    def get_embedding(self, x: torch.Tensor):
        x = F.interpolate(
            x, (self.img_size,)*2,
            mode="bilinear", align_corners=False,
        )
        emb = self.image_encoder(x)
        return nn.functional.adaptive_avg_pool2d(emb, 1).squeeze()




class AutoSamSegGabor2(nn.Module):
    def __init__(
        self,
        image_encoder: nn.Module,
        seg_decoder: nn.Module,
        img_size: int = 1024,
        lambd=16.0
    ):
        super().__init__()
        self.img_size = img_size
        self.image_encoder = image_encoder
        self.mask_decoder = seg_decoder
        self.pe_layer = PositionEmbeddingRandom(128)

        # 4 orientations × 3 scales = 12 filters
        orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        scales       = [8, 16, 32]
        gabor_bank = make_gabor_bank(
            ksize=31,
            sigmas=[0.56 * s for s in scales],
            thetas=orientations,
            lambd=lambd,
        )  # → (12,1,31,31)
        self.register_buffer('gabor_kernels', gabor_bank)



    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        original_size = W

        # 1) resize for encoder
        x = F.interpolate(
            x,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )

        # 2) frozen image features
        with torch.no_grad():
            img_emb = self.image_encoder(x)  # (B,D,H',W')

            # 3) compute texture map
            gray = (0.2989 * x[:,0]
                + 0.5870 * x[:,1]
                + 0.1140 * x[:,2]).unsqueeze(1)  # (B,1,H',W')
            gray = F.interpolate(gray, (256, 256), mode='bilinear', align_corners=False)
            tex = F.conv2d(
                gray,
                self.gabor_kernels,
                padding=self.gabor_kernels.shape[-1]//2
            )  # (B,12,H',W')

        
        # 4) downsample Gabor to match encoder spatial dims
        Hf, Wf = img_emb.shape[-2:]
        gabor_feats = F.interpolate(
            tex,
            (Hf, Wf),
            mode='bilinear',
            align_corners=False)
            

        # 5) positional encoding
        Hf, Wf = img_emb.shape[-2:]
        img_pe = self.pe_layer([Hf, Wf]).unsqueeze(0)  # (1,D,Hf,Wf)

        # 7) decode
        mask, iou_pred = self.mask_decoder(
            image_embeddings=img_emb.unsqueeze(1),
            image_pe=img_pe,
            gabor_feats=gabor_feats
        )

        # 8) resize back
        if mask.shape[-1] != original_size:
            mask = F.interpolate(
                mask, (original_size,)*2,
                mode='bilinear', align_corners=False
            )
        return mask, iou_pred

    def get_embedding(self, x: torch.Tensor):
        x = F.interpolate(
            x, (self.img_size,)*2,
            mode="bilinear", align_corners=False,
        )
        emb = self.image_encoder(x)
        return nn.functional.adaptive_avg_pool2d(emb, 1).squeeze()
    

class AutoSamSegWithDino(nn.Module):
    def __init__(
        self,
        image_encoder: nn.Module,
        seg_decoder: nn.Module,
        dino_encoder: nn.Module,
        img_size: int = 1024,
    ):
        super().__init__()
        self.img_size = img_size
        self.image_encoder = image_encoder
        self.mask_decoder = seg_decoder
        self.pe_layer = PositionEmbeddingRandom(128)
        self.dino_encoder = dino_encoder


        self.alpha = 0.9
     

        # Small conv: 12->out_chans
        out_chans = self.image_encoder.neck[0].out_channelsss
        dino_dim = self.dino_encoder.embed_dim
        self.dino_conv = nn.Sequential(
            nn.Conv2d(dino_dim, out_chans, kernel_size=1, padding=0, bias=False),
            nn.GELU(),
        )


    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        original_size = W

        # 3) DINOv2 inference at 896x896
        x_dino = F.interpolate(x, (896, 896),
                               mode="bilinear", align_corners=False)

        # 1) resize for encoder
        x = F.interpolate(
            x,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )


        # 2) frozen image features
        with torch.no_grad():
            img_emb = self.image_encoder(x)  # (B,D,H',W')

            
        Hf, Wf = img_emb.shape[-2:]

        
        with torch.no_grad():
            feats = self.dino_encoder.forward_features(x_dino)
        patch_tokens = feats['x_norm_patchtokens']  # (B, Np, D)

        # 4) Reshape to 2D grid
        Np = patch_tokens.shape[1]
        # gh = gw = int(Np ** 0.5)
        tokens_2d = patch_tokens.transpose(1, 2).reshape(B, -1, Hf, Wf)  # (B, D, gh, gw)

        # 5) Upsample DINO tokens to SAM feature map
        # tok_up = F.interpolate(tokens_2d, size=(Hf, Wf),
        #                        mode="bilinear", align_corners=False)
       
        dino_emb = self.dino_conv(tokens_2d)  # (B, out_chans, Hf, Wf)

        # Fuse with learnable alpha
        fused = self.alpha * dino_emb + (1 - self.alpha) * img_emb

        # 5) positional encoding
        Hf, Wf = fused.shape[-2:]
        img_pe = self.pe_layer([Hf, Wf]).unsqueeze(0)  # (1,D,Hf,Wf)

        # 7) decode
        mask, iou_pred = self.mask_decoder(
            image_embeddings=fused.unsqueeze(1),
            image_pe=img_pe,
        )

        # 8) resize back
        if mask.shape[-1] != original_size:
            mask = F.interpolate(
                mask, (original_size,)*2,
                mode='bilinear', align_corners=False
            )
        return mask, iou_pred

    def get_embedding(self, x: torch.Tensor):
        x = F.interpolate(
            x, (self.img_size,)*2,
            mode="bilinear", align_corners=False,
        )
        emb = self.image_encoder(x)
        return nn.functional.adaptive_avg_pool2d(emb, 1).squeeze()
    

class MultiScaleFusion(nn.Module):
    def __init__(self, in_chs, out_ch):
        """
        in_chs: list of channel dims for each skip feature
        out_ch: target channel dim (same as neck output → MaskDecoder)
        """
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_c, out_ch, kernel_size=1, bias=False),
                nn.GELU(),
            )
            for in_c in in_chs
        ])

    def forward(self, feats):
        # feats: list of [B, in_c[i], H, W]
        fused = 0
        for proj, feat in zip(self.projs, feats):
            fused = fused + proj(feat)
        return fused
    

class StaticGatedFusion(nn.Module):
    def __init__(self, in_chs, out_ch):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(ic, out_ch, 1, bias=False), nn.GELU())
            for ic in in_chs
        ])
        # one gate per scale, init to 1.0
        self.gates = nn.Parameter(torch.ones(len(in_chs)))
        # one post‐fusion LayerNorm2d
        self.post_norm = LayerNorm2d(out_ch)

    def forward(self, feats, orig_neck=None, use_residual=False):
        # project each scale
        proj_feats = [proj(f) for proj, f in zip(self.projs, feats)]
        # apply sigmoid to keep gates in (0,1)
        gate_vals = torch.sigmoid(self.gates).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        # weighted sum
        fused = sum(g * f for g, f in zip(gate_vals, proj_feats))
        if use_residual and orig_neck is not None:
            fused = fused + orig_neck

        # single normalization after fusion
        fused = self.post_norm(fused)

        return fused

    

class FusionWithPostNorm(nn.Module):
    def __init__(self, in_chs, out_ch):
        super().__init__()
        # your 1×1 projections
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_c, out_ch, kernel_size=1, bias=False),
                nn.GELU(),
            )
            for in_c in in_chs
        ])
        # one post‐fusion LayerNorm2d
        self.post_norm = LayerNorm2d(out_ch)

    def forward(self, feats, orig_neck=None, use_residual=False):
        # feats: list of [B, in_chs[i], H', W']
        fused = 0
        for proj, feat in zip(self.projs, feats):
            fused = fused + proj(feat)

        if use_residual and orig_neck is not None:
            fused = fused + orig_neck

        # single normalization after fusion
        fused = self.post_norm(fused)
        return fused


class AutoSamSegWithFusion(nn.Module):
    def __init__(self,
                 image_encoder,        # your ViT encoder
                 seg_decoder,         # your MaskDecoder
                 fuse_block_indices=[0, 1, 2],
                 residual=False,
                 gated=False,
                 img_size=1024):
        super().__init__()
        self.img_size      = img_size
        self.image_encoder = image_encoder
        self.mask_decoder  = seg_decoder
        self.pe_layer      = PositionEmbeddingRandom(128)

        # how many channels the neck spits out into MaskDecoder:
        out_ch = self.image_encoder.neck[0].out_channels

        # build list of input-channels for each skip + final neck:
        in_chs = []
        for i in fuse_block_indices:
            # each block outputs [B, H, W, embed_dim]
            # its attn.qkv maps from embed_dim → 3*embed_dim, so:
            embed_dim = self.image_encoder.blocks[i].attn.qkv.in_features
            in_chs.append(embed_dim)
        # finally include the neck output channels:
        in_chs.append(out_ch)

        self.fuse_idxs = list(fuse_block_indices)
        if gated:
            self.fuser     = StaticGatedFusion(in_chs, out_ch)
        else:
            self.fuser     = FusionWithPostNorm(in_chs, out_ch)
        self.residual   = residual

    def forward(self, x):
        B, _, H0, W0 = x.shape

        # 1) resize
        x = F.interpolate(x,
                          (self.img_size, self.img_size),
                          mode='bilinear', align_corners=False)

        # 2) patch‐embed + abs‐pos
        x = self.image_encoder.patch_embed(x)  # [B, Hp, Wp, C]
        if self.image_encoder.pos_embed is not None:
            x = x + self.image_encoder.pos_embed

        # 3) run through blocks & collect skips
        feats = []
        for idx, blk in enumerate(self.image_encoder.blocks):
            x = blk(x)  # [B, Hp, Wp, C]
            if idx in self.fuse_idxs:
                # project to [B, C, Hp, Wp]
                feats.append(x.permute(0,3,1,2))

        # 4) neck on the *last* x
        x_neck = self.image_encoder.neck(x.permute(0,3,1,2))  # [B, out_ch, H′, W′]
        feats.append(x_neck)

        # 5) fuse them all
        fused = self.fuser(feats, orig_neck=x_neck, use_residual=self.residual)  # [B, out_ch, H′, W′]


        # 6) positional encoding + mask decode
        Hf, Wf = fused.shape[-2:]
        img_pe = self.pe_layer([Hf, Wf]).unsqueeze(0)
        masks, iou = self.mask_decoder(
            image_embeddings=fused.unsqueeze(1),
            image_pe=img_pe,
        )

        # 7) back to original
        if masks.shape[-1] != W0:
            masks = F.interpolate(masks,
                                  (W0, W0),
                                  mode='bilinear', align_corners=False)
        return masks, iou



class AutoSamSeg(nn.Module):
    def __init__(
        self,
        image_encoder,
        seg_decoder,
        img_size=1024,
    ):
        super().__init__()
        self.img_size = img_size
        self.image_encoder = image_encoder
        self.mask_decoder = seg_decoder
        self.pe_layer = PositionEmbeddingRandom(128)

    def forward(self,
                x):
        original_size = x.shape[-1]
        x = F.interpolate(
            x,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        image_embedding = self.image_encoder(x) #[B, 256, 64, 64]
        img_pe = self.pe_layer([64, 64]).unsqueeze(0)
        mask, iou_pred = self.mask_decoder(image_embeddings=image_embedding.unsqueeze(1),
                                           image_pe=img_pe, )

        if mask.shape[-1] != original_size:
            mask = F.interpolate(
                mask,
                (original_size, original_size),
                mode="bilinear",
                align_corners=False,
            )
        return mask, iou_pred

    def get_embedding(self, x):
        original_size = x.shape[-1]
        x = F.interpolate(
            x,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        image_embedding = self.image_encoder(x)
        out = nn.functional.adaptive_avg_pool2d(image_embedding, 1).squeeze()
        return out
