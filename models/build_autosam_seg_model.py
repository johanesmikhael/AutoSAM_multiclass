import torch

from functools import partial

from .SamFeatSeg import SamFeatSeg, SegDecoderCNN
from .AutoSamSeg import AutoSamSeg
from .sam_decoder import MaskDecoder
from segment_anything.modeling import ImageEncoderViT, TwoWayTransformer


def _build_sam_seg_model(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    num_classes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    sam_seg = AutoSamSeg(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        seg_decoder=MaskDecoder(
            num_multimask_outputs=1,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            num_classes=num_classes,
        ),
    )

    if checkpoint is not None:
        # 1) Load the full checkpoint dict
        ckpt = torch.load(checkpoint)
        # (or ckpt = ckpt["model"] if your file nests it)

        # 2) Split into two dicts:
        #    - encoder_ckpt: keys for image_encoder.*, stripped of that prefix
        #    - rest_ckpt: all the other keys you want to load non‐strictly
        encoder_ckpt = {}
        rest_ckpt = {}
        for k, v in ckpt.items():
            if not k in sam_seg.state_dict():
                continue
            # pick off your encoder weights
            if k.startswith("image_encoder."):
                # strip the module prefix so it matches ImageEncoderViT.state_dict()
                stripped = k[len("image_encoder."):]
                encoder_ckpt[stripped] = v
            else:
                # filter out IOU heads, mask tokens, prompt_encoder, etc.
                if (
                    'iou' not in k
                    and 'mask_tokens' not in k
                    and not k.startswith("prompt_encoder.")
                ):
                    rest_ckpt[k] = v

        # 3) Strict‐load the encoder
        missing_e, unexpected_e = sam_seg.image_encoder.load_state_dict(
            encoder_ckpt, strict=True
        )
        print("ImageEncoder missing keys:", missing_e)
        print("ImageEncoder unexpected keys:", unexpected_e)

        # 4) Non‐strict‐load the rest into the full SAM model
        missing_r, unexpected_r = sam_seg.load_state_dict(
            rest_ckpt, strict=False
        )
        print("Rest (decoder, etc.) missing keys:", missing_r)
        print("Rest unexpected keys:", unexpected_r)

    return sam_seg


def build_sam_vit_h_seg_cnn(num_classes=14, checkpoint=None):
    return _build_sam_seg_model(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        num_classes=num_classes,
        checkpoint=checkpoint,
    )


build_sam_seg = build_sam_vit_h_seg_cnn


def build_sam_vit_l_seg_cnn(num_classes=14, checkpoint=None):
    return _build_sam_seg_model(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        num_classes=num_classes,
        checkpoint=checkpoint,
    )


def build_sam_vit_b_seg_cnn(num_classes=14, checkpoint=None):
    return _build_sam_seg_model(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        num_classes=num_classes,
        checkpoint=checkpoint,
    )


sam_seg_model_registry = {
    "default": build_sam_seg,
    "vit_h": build_sam_seg,
    "vit_l": build_sam_vit_l_seg_cnn,
    "vit_b": build_sam_vit_b_seg_cnn,
}

