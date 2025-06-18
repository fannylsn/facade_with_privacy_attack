# import os
# from sklearn.model_selection import train_test_split


# def split_dataset_with_links(base_dir, train_size=0.8):
#     classes = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
#     for cls in classes:
#         class_dir = os.path.join(base_dir, cls)
#         subfolders = [d for d in os.listdir(class_dir) if os.path.isdir(os.path.join(class_dir, d))]

#         # Split subfolders into training and testing
#         train_subfolders, test_subfolders = train_test_split(subfolders, train_size=train_size, random_state=42)

#         # Create links for train and test
#         def create_links(subfolders, destination):
#             dest_class_dir = os.path.join(base_dir, destination, cls)
#             os.makedirs(dest_class_dir, exist_ok=True)
#             for subfolder in subfolders:
#                 src_path = os.path.join(class_dir, subfolder)
#                 dest_path = os.path.join(dest_class_dir, subfolder)
#                 os.symlink(src_path, dest_path)

#         # Create symbolic links in respective directories
#         create_links(train_subfolders, "train")
#         create_links(test_subfolders, "test")


# base_dir = "./eval/data/geo_animal/images"  # Adjust this to your actual base directory
# split_dataset_with_links(base_dir)

import os

from sklearn.model_selection import train_test_split


def create_train_test_splits(base_dir, train_size=0.8):
    """
    Create train/test splits for the dataset and flatten the structure with symlinks.
    """
    images_dir = os.path.join(base_dir, "images")
    train_dir = os.path.join(base_dir, "train2")
    test_dir = os.path.join(base_dir, "test2")

    # Create the train and test directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get all class directories from the images folder
    classes = [
        d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))
    ]

    for cls in classes:
        class_path = os.path.join(images_dir, cls)
        subfolders = [
            os.path.join(class_path, d)
            for d in os.listdir(class_path)
            if os.path.isdir(os.path.join(class_path, d))
        ]

        # Split subfolders into train and test
        train_subfolders, test_subfolders = train_test_split(
            subfolders, train_size=train_size, random_state=42
        )

        # Function to create symlinks for images in subfolders to flatten the structure
        def create_symlinks_files(subfolders, dest_dir):
            dest_class_dir = os.path.join(dest_dir, cls)
            os.makedirs(dest_class_dir, exist_ok=True)
            for subfolder in subfolders:
                for img_file in os.listdir(subfolder):
                    src_img_path = os.path.join(subfolder, img_file)
                    dest_img_path = os.path.join(dest_class_dir, img_file)
                    # Create a symlink in the destination directory pointing to the source image
                    os.symlink(src_img_path, dest_img_path)

        def create_symlinks(subfolders, dest_dir):
            dest_class_dir = os.path.join(dest_dir, cls)
            os.makedirs(dest_class_dir, exist_ok=True)
            for subfolder in subfolders:
                dest_path = os.path.join(dest_class_dir, os.path.basename(subfolder))
                # Create a symlink in the destination directory pointing to the source directory
                os.symlink(subfolder, dest_path, target_is_directory=True)

        # Create symlinks for train and test datasets
        create_symlinks(train_subfolders, train_dir)
        create_symlinks(test_subfolders, test_dir)


# Example usage
base_dir = "/mnt/nfs/thiba/decentralizepy/eval/data/geo_animal"  # must be absolute !
create_train_test_splits(base_dir)
