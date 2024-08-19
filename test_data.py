import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data import DepthToSketchDataset


def visualize_sample(sample, index):
    depth_image = sample['depth_image'].numpy().transpose(1, 2, 0)
    sketch_image = sample['sketch_image'].numpy().transpose(1, 2, 0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(depth_image)
    axes[0].set_title('Depth Image')
    axes[0].axis('off')

    axes[1].imshow(sketch_image, cmap='gray')
    axes[1].set_title('Sketch Image')
    axes[1].axis('off')

    plt.suptitle(f"Sample {index}")
    plt.show()

def test_dataloader():
    path_json = "path/to/your/json_file.json"  # Replace with the actual path to your JSON file
    path_meta = "path/to/your/data_root_folder"  # Replace with the actual path to your data root folder

    # Instantiate the dataset
    dataset = DepthToSketchDataset(path_json=path_json, path_meta=path_meta, resolution=256)

    # Create a dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Iterate over a few samples and visualize them
    for idx, sample in enumerate(dataloader):
        print(f"Testing sample {idx + 1}/{len(dataset)}")
        visualize_sample(sample, idx + 1)

        # Optionally, stop after a few samples
        if idx >= 4:
            break

if __name__ == "__main__":
    test_dataloader()
