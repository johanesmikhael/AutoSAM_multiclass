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
            lambd=16.0,
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


        # 2) frozen image features
        with torch.no_grad():
            img_emb = self.image_encoder(x)  # (B,D,H',W')

            # 3) compute texture map
            gray = (0.2989 * x[:,0]
                + 0.5870 * x[:,1]
                + 0.1140 * x[:,2]).unsqueeze(1)  # (B,1,H',W')
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
            lambd=16.0,
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
