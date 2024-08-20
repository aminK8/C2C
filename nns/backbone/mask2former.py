import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmengine import Config
from mmseg.apis import init_model
from mmseg.models.utils import resize
from typing import Tuple


class Mask2Former(nn.Module):
    def __init__(self, cfg):
        super(Mask2Former, self).__init__()

        mmcfg = Config.fromfile(cfg['config_file'])

        pre_model = init_model(mmcfg, cfg['pretrained_checkpoint_file'], device='cpu')

        self.backbone = pre_model.backbone
        self.decode_head = pre_model.decode_head
        self.img_shape = cfg['image_size']

    def _forward_head(self, decoder_out: Tensor, mask_feature: Tensor,
                      attn_mask_target_size: Tuple[int, int]) -> Tuple[Tensor]:
        """Forward for head part which is called after every decoder layer.

        Args:
            decoder_out (Tensor): in shape (batch_size, num_queries, c).
            mask_feature (Tensor): in shape (batch_size, c, h, w).
            attn_mask_target_size (tuple[int, int]): target attention
                mask size.

        Returns:
            tuple: A tuple contain three elements.

                - cls_pred (Tensor): Classification scores in shape \
                    (batch_size, num_queries, cls_out_channels). \
                    Note `cls_out_channels` should includes background.
                - mask_pred (Tensor): Mask scores in shape \
                    (batch_size, num_queries,h, w).
                - attn_mask (Tensor): Attention mask in shape \
                    (batch_size * num_heads, num_queries, h, w).
        """
        # Modified from: https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/dense_heads/mask2former_head.py

        decoder_out = self.decode_head.transformer_decoder.post_norm(decoder_out)
        # shape (num_queries, batch_size, c)
        # cls_pred = self.cls_embed(decoder_out)
        # shape (num_queries, batch_size, c)
        mask_embed = self.decode_head.mask_embed(decoder_out)
        # shape (num_queries, batch_size, h, w)
        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)
        attn_mask = F.interpolate(
            mask_pred,
            attn_mask_target_size,
            mode='bilinear',
            align_corners=False)
        # shape (num_queries, batch_size, h, w) ->
        #   (batch_size * num_head, num_queries, h, w)
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat(
            (1, self.decode_head.num_heads, 1, 1)).flatten(0, 1)
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()

        return decoder_out, mask_pred, attn_mask

    def forward(self, x):
        backbone_output = self.backbone(x)
        # output shapes:
        # (b, 128, 160, 160)
        # (b, 256, 80, 80)
        # (b, 512, 40, 40)
        # (b, 1024, 20, 20)

        # Modified from: https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/models/decode_heads/mask2former_head.py
        # https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/dense_heads/mask2former_head.py

        batch_size = backbone_output[0].shape[0]
        mask_features, multi_scale_memorys = self.decode_head.pixel_decoder(backbone_output)
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.decode_head.num_transformer_feat_level):
            decoder_input = self.decode_head.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.decode_head.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            mask = decoder_input.new_zeros(
                (batch_size, ) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decode_head.decoder_positional_encoding(
                mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (batch_size, num_queries, c)
        query_feat = self.decode_head.query_feat.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))
        query_embed = self.decode_head.query_embed.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))

        # cls_pred_list = []
        # mask_pred_list = []
        cls_pred, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
        # cls_pred_list.append(cls_pred)
        # mask_pred_list.append(mask_pred)

        for i in range(self.decode_head.num_transformer_decoder_layers):
            level_idx = i % self.decode_head.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            mask_sum = (attn_mask.sum(-1) != attn_mask.shape[-1]).unsqueeze(-1)
            attn_mask = attn_mask & mask_sum
            # cross_attn + self_attn
            layer = self.decode_head.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)
            cls_pred, mask_pred, attn_mask = self._forward_head(
                query_feat, mask_features, multi_scale_memorys[(i + 1) % self.decode_head.num_transformer_feat_level].shape[-2:])

            # cls_pred_list.append(cls_pred)
            # mask_pred_list.append(mask_pred)

        # mask_cls_results = cls_pred_list[-1]
        # mask_pred_results = mask_pred_list[-1]
        # mask_pred_results = F.interpolate(
        #     mask_pred_results, size=self.img_shape, mode='bilinear', align_corners=False)
        # cls_score = F.softmax(mask_cls_results, dim=-1)[..., :-1]
        # mask_pred = mask_pred_results.sigmoid()
        # seg_logits = torch.einsum('bqc, bqhw->bchw', cls_score, mask_pred)

        # cls_pred shape: (b, 100, 256)
        # mask_pred shape: (b, 100, 160, 160)
        return mask_pred, backbone_output[3], cls_pred


if __name__ == "__main__":
    import json, argparse
    with open('cfg/model/mask2former.json', 'r') as f:
        m_cfg = json.loads(f.read())
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--local_rank', default=0, type=int
    )
    args = parser.parse_args()
    model = Mask2Former(args, m_cfg['model'])
    input_tensor = torch.randn(1, 3, m_cfg['processor']['image_size'][0], m_cfg['processor']['image_size'][1])
    cls_pred, mask_pred = model(input_tensor)

    # from mmseg.apis import inference_model, show_result_pyplot
    # img_path = 'samples/image1.jpg'
    # cfg = m_cfg['model']
    # model = init_model(cfg['config_file'], cfg['pretrained_checkpoint_file'], device='cpu')
    # result = inference_model(model, img_path)
    # vis_image = show_result_pyplot(model, img_path, result)
