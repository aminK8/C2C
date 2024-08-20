import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


class SegAnyModel(nn.Module):
    def __init__(self, cfg):
        super(SegAnyModel, self).__init__()

        model_type = cfg['name'][4:]  # e.g. "vit_b", "vit_h"
        pre_model = sam_model_registry[model_type](checkpoint=cfg['pretrained_checkpoint_file'])

        self.image_encoder = pre_model.image_encoder
        self.mask_decoder = pre_model.mask_decoder
        self.prompt_encoder = pre_model.prompt_encoder

        self.multimask_output = cfg['multimask_output']

    def forward(self, x):
        image_embedding = self.image_encoder(x)  # (B, 256, 64, 64)

        # generate dummy prompt embeddings for mask_decoder
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )

        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=self.multimask_output,
        )
        # low_res_masks: (B, 3, 256, 256)
        # or (B, 1, 256, 256) when multimask_output=False

        return low_res_masks, image_embedding


if __name__ == "__main__":
    import json
    with open('cfg/model/sam_vit_b.json', 'r') as f:
        m_cfg = json.loads(f.read())
    model = SegAnyModel(m_cfg['model'])
    params_count = sum(p.numel() for p in model.parameters())
    print(f'Num params: {params_count}')
    
    input_tensor = torch.randn(1, 3, m_cfg['processor']['image_size'][0], m_cfg['processor']['image_size'][1])
    output = model(input_tensor)
