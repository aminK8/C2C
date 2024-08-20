import torch
import torch.nn as nn
from mmengine import Config
from mmseg.apis import init_model
from mmseg.models.utils import resize


class Segformer(nn.Module):
    def __init__(self, cfg):
        super(Segformer, self).__init__()

        mmcfg = Config.fromfile(cfg['config_file'])

        pre_model = init_model(mmcfg, cfg['pretrained_checkpoint_file'], device='cpu')

        self.backbone = pre_model.backbone
        self.decode_head = pre_model.decode_head

    def forward(self, x):
        backbone_output = self.backbone(x)
        # output shapes:
        # (b, 64, 128, 128)
        # (b, 128, 64, 64)
        # (b, 320, 32, 32)
        # (b, 512, 16, 16)

        # Taken from: https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/models/decode_heads/segformer_head.py
        inputs = self.decode_head._transform_inputs(backbone_output)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.decode_head.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.decode_head.interpolate_mode,
                    align_corners=self.decode_head.align_corners))

        decode_head_output = self.decode_head.fusion_conv(torch.cat(outs, dim=1))
        # output shape:
        # (b, 256, 128, 128)
        return decode_head_output, backbone_output[3]


if __name__ == "__main__":
    import json
    with open('cfg/model/segformer.json', 'r') as f:
        m_cfg = json.loads(f.read())
    model = Segformer(m_cfg['model'])
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
