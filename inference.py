import os
import tqdm
import torch
import cv2
import numpy as np
import argparse
import json
from torch.utils.data import DataLoader, random_split
from data import DepthToSketchDataset, CondCvtrTransform
from nns.model import CondCvtr
from utils import print_main, fix_random_seeds


def get_args_parser():
    parser = argparse.ArgumentParser('Spectrum-OCT', add_help=False)

    # File paths
    parser.add_argument('--config_file', default='cfg/segformer.json', type=str,
        help='Path to the model configuration file.')

    parser.add_argument('--checkpoint_dir', default='runs/segformer_exp0', type=str, help='Path to the checkpoint directory.')
    parser.add_argument('--load_ckpt_id', default='last', type=str, help='Checkpoint ID.')
    parser.add_argument('--save_dir', default='tests/segformer_exp0', type=str, help='Path to the save directory.')
    parser.add_argument('--seed', default=42, type=int, help='Random seed.')
    parser.add_argument('--train_ratio', default=0.9, type=float, help='Ratio of training set to the whole dataset.')
    return parser


def inference(args):
    fix_random_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    with open(args.config_file, 'r') as f:
        cfg = json.loads(f.read())
    print_main("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    print_main("\n".join("%s: %s" % (k, str(v)) for k, v in cfg.items()))

    # Instantiate the dataset
    dataset = DepthToSketchDataset(path_json_depth=cfg['data']['path_json_depth'],
                                   path_json_sketch=cfg['data']['path_json_sketch'], 
                                   path_meta=cfg['data']['path_meta'],
                                   resolution=cfg['processor']['resize'][0])
    dataset.transform = CondCvtrTransform(cfg=cfg['processor'])

    train_size = int(args.train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f'Train dataset samples: {len(train_dataset)}')
    print(f'Train dataset samples: {len(val_dataset)}')

    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = CondCvtr(cfg).to(device)
    ckpt_file = os.path.join(args.checkpoint_dir, f'model_{args.load_ckpt_id}.pt')
    model.load_state_dict(torch.load(ckpt_file, map_location=device))

    # Start testing
    model.eval()
    for batch in tqdm.tqdm(val_dataloader):
        im_file = os.path.join(args.save_dir, batch["filename_depth"][0].split('/')[-1])
        if os.path.exists(im_file):
            continue

        inputs = batch['depth_image'].to(device)
        labels = batch['sketch_image']
        outputs = model(inputs)

        inputs = inputs[0].detach().cpu().numpy()
        inputs = inputs * np.array(cfg['processor']['image_std'])[:, np.newaxis, np.newaxis]\
            + np.array(cfg['processor']['image_mean'])[:, np.newaxis, np.newaxis]
        inputs *= 255
        inputs = np.clip(inputs, 0, 255).astype(np.uint8)
        inputs = inputs.transpose(1, 2, 0)

        outputs = outputs[0].detach().cpu().numpy()
        outputs = np.argmax(outputs, axis=0).astype(np.uint8)
        outputs = np.repeat(outputs[:, :, np.newaxis], 3, axis=2)

        labels = labels[0].numpy().astype(np.uint8)
        labels = np.repeat(labels[:, :, np.newaxis], 3, axis=2)

        concat_img = np.concatenate((inputs, labels, outputs), axis=1)
        cv2.imwrite(im_file, concat_img)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    inference(args)
