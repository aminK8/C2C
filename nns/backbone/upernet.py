import torch
import torch.nn as nn
from mmengine import Config
from mmseg.apis import init_model


class UperNet(nn.Module):
    def __init__(self, cfg):
        super(UperNet, self).__init__()

        mmcfg = Config.fromfile(cfg['config_file'])

        pre_model = init_model(mmcfg, cfg['pretrained_checkpoint_file'], device='cpu')

        self.backbone = pre_model.backbone
        self.decode_head = pre_model.decode_head
        self.auxiliary_head = pre_model.auxiliary_head

    def forward(self, x):
        backbone_output = self.backbone(x)
        # backbone_output shapes:
        # ResNet101: (b, 256, 128, 128), (b, 512, 64, 64), (b, 1024, 32, 32), (b, 2048, 16, 16)
        # Swin-B4: (b, 128, 128, 128), (b, 256, 64, 64), (b, 512, 32, 32), (b, 1024, 16, 16)
        # ConvNeXt-L: (b, 192, 160, 160), (b, 384, 80, 80), (b, 768, 40, 40), (b, 1536, 20, 20)

        decode_head_output = self.decode_head._forward_feature(backbone_output)
        # decode_head_output shape:
        # ResNet101: (b, 512, 128, 128)
        # Swin-B4: (b, 512, 128, 128)
        # ConvNeXt-L: (b, 512, 160, 160)

        # auxiliary_head_output = self.auxiliary_head._forward_feature(backbone_output)
        # auxiliary_head_output shape:
        # ResNet101: (b, 256, 32, 32)
        # Swin-B4: (b, 256, 32, 32)
        # ConvNeXt-L: (b, 256, 40, 40)

        return decode_head_output, backbone_output[3]


if __name__ == "__main__":
    import json
    # config_file = 'cfg/model/upernet_r101.json'
    config_file = 'cfg/model/upernet_swin_b4.json'
    # config_file = 'cfg/model/upernet_convnext.json'
    with open(config_file, 'r') as f:
        m_cfg = json.loads(f.read())

    model = UperNet(m_cfg['model'])
    input_tensor = torch.randn(1, 3, m_cfg['processor']['image_size'][0], m_cfg['processor']['image_size'][1])
    output = model(input_tensor)
    print(output[0].shape)
    print(output[1].shape)

    # from mmseg.apis import inference_model, show_result_pyplot
    # img_path = 'samples/image1.jpg'
    # cfg = m_cfg['model']
    # model = init_model(cfg['config_file'], cfg['pretrained_checkpoint_file'], device=f'cuda:{args.local_rank}')
    # result = inference_model(model, img_path)
    # vis_image = show_result_pyplot(model, img_path, result)
