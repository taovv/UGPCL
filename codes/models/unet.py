from ._base import BaseModel2D
from typing import Optional, Union, List
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import SegmentationModel, SegmentationHead, ClassificationHead
from segmentation_models_pytorch.unet.decoder import UnetDecoder


class Unet(SegmentationModel):

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def forward_features(self, x):
        features = self.encoder(x)
        return features


class UNet(BaseModel2D):

    def __init__(self,
                 encoder_name="resnet34",
                 encoder_depth=5,
                 encoder_weights="imagenet",
                 decoder_use_batchnorm: bool = True,
                 decoder_channels=(256, 128, 64, 32, 16),
                 decoder_attention_type=None,
                 in_channels=3,
                 classes=1,
                 activation=None,
                 aux_params=None):
        super().__init__()

        self.segmentor = Unet(encoder_name=encoder_name,
                              encoder_depth=encoder_depth,
                              encoder_weights=encoder_weights,
                              decoder_use_batchnorm=decoder_use_batchnorm,
                              decoder_channels=decoder_channels,
                              decoder_attention_type=decoder_attention_type,
                              in_channels=in_channels,
                              classes=classes,
                              activation=activation,
                              aux_params=aux_params)
        self.num_classes = classes

    def forward(self, x):
        return {'seg': self.segmentor(x)}

    def inference_features(self, x, **kwargs):
        feats = self.segmentor.forward_features(x)
        return {'feats': feats}
