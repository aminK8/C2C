import json
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data import DepthToSketchDataset, CondCvtrTransform


def visualize_sample(sample, index):
    depth_image = sample['depth_image'].numpy()[0].transpose(1, 2, 0)
    sketch_image = sample['sketch_image'].numpy()[0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(depth_image)
    axes[0].set_title('Depth Image')
    axes[0].axis('off')

    axes[1].imshow(sketch_image, cmap='gray')
    axes[1].set_title('Sketch Image')
    axes[1].axis('off')

    # plt.suptitle(f"Sample {index}")
    # plt.show()

def test_dataloader():
    with open('cfg/segformer.json', 'r') as f:
        cfg = json.loads(f.read())

    # Instantiate the dataset
    dataset = DepthToSketchDataset(path_json_depth=cfg['data']['path_json_depth'],
                                   path_json_sketch=cfg['data']['path_json_sketch'], 
                                   path_meta=cfg['data']['path_meta'],
                                   resolution=cfg['processor']['resize'][0])
    dataset.transform = CondCvtrTransform(cfg=cfg['processor'])

    # Create a dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Iterate over a few samples and visualize them
    for idx, sample in enumerate(dataloader):
        print(f"Testing sample {idx + 1}/{len(dataset)}")
        # visualize_sample(sample, idx + 1)
        dimg = sample['depth_image'][0].numpy().transpose(1, 2, 0)
        simg = sample['sketch_image'][0].numpy()

        # Optionally, stop after a few samples
        if idx >= 4:
            break

if __name__ == "__main__":
    test_dataloader()
