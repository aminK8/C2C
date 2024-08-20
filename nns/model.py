import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import Tuple, Union
from torch import Tensor
from nns.backbone import UperNet, Mask2Former, Segformer


class CondCvtr(nn.Module):
    def __init__(self,
                 cfg: dict):
        super(CondCvtr, self).__init__()

        model_cfg = cfg['model']
        self.enc_dec_name = model_cfg['name'].lower()
        if self.enc_dec_name == 'segformer':
            self.enc_dec_model = Segformer(model_cfg)
        elif self.enc_dec_name == 'mask2former':
            self.enc_dec_model = Mask2Former(model_cfg)
        elif self.enc_dec_name.startswith('upernet'):
            self.enc_dec_model = UperNet(model_cfg)
        else:
            ValueError('Model type not supported: {}'.format(model_cfg['name']))

        self.img_size = cfg['processor']['resize']

        self.out_proj = nn.Conv2d(model_cfg['encdec_out_channels'],
                                  model_cfg['num_seg_classes'],
                                  kernel_size=1)
        
        self._init_params()

    def _init_params(self):
        # seg head
        init.kaiming_normal_(self.out_proj.weight, mode='fan_out', nonlinearity='relu')
        init.constant_(self.out_proj.bias, 0)

    def freeze_pretrained_model(self,
                                mode: str = 'enc_dec'):
        if mode == 'enc_dec':
            for p in self.enc_dec_model.parameters():
                p.requires_grad_(False)
        elif mode == 'backbone':
            for p in self.enc_dec_model.backbone.parameters():
                p.requires_grad_(False)
        else:
            ValueError(f'\"mode\" needs to be within [\"enc_dec\", \"backbone\"].')
    
    def forward(self,
                x: Tensor
                ) -> Tuple[Tensor, Tensor, Tensor]:
        enc_dec_output = self.enc_dec_model(x)  # (b, c, h, w)

        decoder_output = enc_dec_output[0]
        # encoder_output = enc_dec_output[1]

        out = self.out_proj(decoder_output)  # (b, num_class, h, w)
        out = F.interpolate(out, size=self.img_size, mode='bilinear', align_corners=False)
        return out


if __name__ == "__main__":
    import json
    with open('cfg/segformer.json', 'r') as f:
        m_cfg = json.loads(f.read())

    model = CondCvtr(m_cfg)
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_params_count = sum(p.numel() for p in trainable_params)
    freezed_params_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f'Num trainable params: {trainable_params_count}')
    print(f'Num freezed params: {freezed_params_count}')

    