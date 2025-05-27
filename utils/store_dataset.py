import os
import shutil


def store_dataset(path, labels_path, source_path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('lesion.bmp'):
                print(f"Copying lesion file: {file}")
                src = os.path.join(root, file)
                dst = os.path.join(labels_path, file)
                shutil.copy2(src, dst)
            if 'Dermoscopic_Image' in root and file.endswith('.bmp'):
                print(f"Copying dermoscopic image file: {file}")
                src = os.path.join(root, file)
                dst = os.path.join(source_path, file)
                shutil.copy2(src, dst)
    return

if __name__ == "__main__":
    store_dataset("/home/karljoker/codeProjects/unet_segmentation/data/raw/PH2 Dataset images", "/home/karljoker/codeProjects/unet_segmentation/data/processed/labels", "/home/karljoker/codeProjects/unet_segmentation/data/processed/source")