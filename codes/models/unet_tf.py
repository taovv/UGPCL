import torch
import torch.nn as nn
import torch.nn.functional as F

from codes.models.swin_decoder import SwinTransDecoder
from codes.models._base import BaseModel2D
from codes.utils.init import kaiming_normal_init_weight
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base import SegmentationHead, ClassificationHead


class EmbeddingHead(nn.Module):
    def __init__(self, dim_in, embed_dim=256, embed='convmlp'):
        super(EmbeddingHead, self).__init__()

        if embed == 'linear':
            self.embed = nn.Conv2d(dim_in, embed_dim, kernel_size=1)
        elif embed == 'convmlp':
            self.embed = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=1),
                nn.BatchNorm2d(dim_in),
                nn.ReLU(),
                nn.Conv2d(dim_in, embed_dim, kernel_size=1)
            )

    def forward(self, x):
        return F.normalize(self.embed(x), p=2, dim=1)


class UNetTF(BaseModel2D):

    def __init__(self,
                 encoder_name="resnet50",
                 encoder_depth=5,
                 encoder_weights="imagenet",
                 decoder_use_batchnorm=True,
                 decoder_channels=(256, 128, 64, 32, 16),
                 decoder_attention_type=None,
                 in_channels=3,
                 classes=2,
                 activation=None,
                 embed_dim=96,
                 norm_layer=nn.LayerNorm,
                 img_size=224,
                 patch_size=4,
                 depths=[2, 2, 2, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 use_checkpoint=False,
                 ape=True,
                 cls=True,
                 contrast_embed=False,
                 contrast_embed_dim=256,
                 contrast_embed_index=-3,
                 mlp_ratio=4.,
                 drop_path_rate=0.1,
                 final_upsample="expand_first",
                 patches_resolution=[56, 56]
                 ):
        super().__init__()
        self.cls = cls
        self.contrast_embed_index = contrast_embed_index

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        encoder_channels = self.encoder.out_channels
        self.cnn_decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )
        self.seg_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )
        self.swin_decoder = SwinTransDecoder(classes, embed_dim, norm_layer, img_size, patch_size, depths, num_heads,
                                             window_size, qkv_bias, qk_scale, drop_rate, attn_drop_rate, use_checkpoint,
                                             ape, mlp_ratio, drop_path_rate, final_upsample, patches_resolution,
                                             encoder_channels)

        self.cls_head = ClassificationHead(in_channels=encoder_channels[-1], classes=4) if cls else None
        self.embed_head = EmbeddingHead(dim_in=encoder_channels[contrast_embed_index],
                                        embed_dim=contrast_embed_dim) if contrast_embed else None
        self._init_weights()

    def _init_weights(self):
        kaiming_normal_init_weight(self.cnn_decoder)
        kaiming_normal_init_weight(self.seg_head)
        if self.cls_head is not None:
            kaiming_normal_init_weight(self.cls_head)
        if self.embed_head is not None:
            kaiming_normal_init_weight(self.embed_head.embed)

    def forward(self, x, device):
        features = self.encoder(x)
        seg = self.seg_head(self.cnn_decoder(*features))
        seg_tf = self.swin_decoder(features, device)

        embedding = self.embed_head(features[self.contrast_embed_index]) if self.embed_head else None
        cls = self.cls_head(features[-1]) if self.cls_head else None
        return {'seg': seg, 'seg_tf': seg_tf, 'cls': cls, 'embed': embedding}

    def inference(self, x, **kwargs):
        features = self.encoder(x)
        seg = self.seg_head(self.cnn_decoder(*features))
        preds = torch.argmax(seg, dim=1, keepdim=True).to(torch.float)
        return preds

    def inference_features(self, x, **kwargs):
        features = self.encoder(x)
        embedding = self.embed_head(features[self.contrast_embed_index]) if self.embed_head else None
        return {'feats': features, 'embed': embedding}

    def inference_tf(self, x, device, **kwargs):
        features = self.encoder(x)
        seg_tf = self.swin_decoder(features, device)
        preds = torch.argmax(seg_tf, dim=1, keepdim=True).to(torch.float)
        return preds
