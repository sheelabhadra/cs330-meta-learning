import numpy as np
import os
import random
import torch


def get_images(paths, labels, nb_samples=None, shuffle=True):
    """
    Takes a set of character folders and labels and returns paths to image files
    paired with labels.
    Args:
        paths: A list of character folders
        labels: List or numpy array of same length as paths
        nb_samples: Number of images to retrieve per character
    Returns:
        List of (label, image_path) tuples
    """
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images_labels = [
        (i, os.path.join(path, image))
        for i, path in zip(labels, paths)
        for image in sampler(os.listdir(path))
    ]
    if shuffle:
        random.shuffle(images_labels)
    return images_labels


def image_file_to_array(filename, dim_input):
    """
    Takes an image path and returns numpy array
    Args:
        filename: Image filename
        dim_input: Flattened shape of image
    Returns:
        1 channel image
    """
    import imageio

    image = imageio.imread(filename)
    image = image.reshape([dim_input])
    image = image.astype(np.float32) / 255.0
    image = 1.0 - image
    return image


class DataGenerator(object):
    """
    Data Generator capable of generating batches of Omniglot data.
    A "class" is considered a class of omniglot digits.
    """

    def __init__(
        self, num_classes, num_samples_per_class, config={}, device=torch.device("cpu")
    ):
        """
        Args:
            num_classes: int
                Number of classes for classification (N-way)

            num_samples_per_class: int
                Number of samples per class in the support set (K-shot).
                Will generate additional sample for the querry set.

            device: cuda.device:
                Device to allocate tensors to.
        """
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes

        data_folder = config.get("data_folder", "./omniglot_resized")
        self.img_size = config.get("img_size", (28, 28))

        self.dim_input = np.prod(self.img_size)
        self.dim_output = self.num_classes

        character_folders = [
            os.path.join(data_folder, family, character)
            for family in os.listdir(data_folder)
            if os.path.isdir(os.path.join(data_folder, family))
            for character in os.listdir(os.path.join(data_folder, family))
            if os.path.isdir(os.path.join(data_folder, family, character))
        ]

        random.seed(1)
        random.shuffle(character_folders)
        num_val = 100
        num_train = 1100
        self.metatrain_character_folders = character_folders[:num_train]
        self.metaval_character_folders = character_folders[
            num_train : num_train + num_val
        ]
        self.metatest_character_folders = character_folders[num_train + num_val :]
        self.device = device

    def sample_batch(self, batch_type, batch_size):
        """
        Samples a batch for training, validation, or testing
        Args:
            batch_type: str
                train/val/test set to sample from

            batch_size: int:
                Size of batch of tasks to sample

        Returns:
            images: tensor
                A tensor of images of size [B, K+1, N, 784]
                where B is batch size, K is number of samples per class,
                N is number of classes

            labels: tensor
                A tensor of images of size [B, K+1, N, N]
                where B is batch size, K is number of samples per class,
                N is number of classes
        """
        if batch_type == "train":
            folders = self.metatrain_character_folders
        elif batch_type == "val":
            folders = self.metaval_character_folders
        else:
            folders = self.metatest_character_folders

        #############################
        #### YOUR CODE GOES HERE ####
        #############################

        # SOLUTION:
        labels = np.arange(self.num_classes)
        labels_one_hot = np.eye(np.max(labels) + 1)[labels]

        imgs = np.zeros(
            (
                batch_size,
                self.num_samples_per_class + 1,
                self.num_classes,
                self.dim_input,
            )
        )
        labels = np.zeros(
            (
                batch_size,
                self.num_samples_per_class + 1,
                self.num_classes,
                self.dim_output,
            )
        )

        for task_num in range(batch_size):
            data_unordered = get_images(
                folders, labels_one_hot, self.num_samples_per_class + 1, False
            )

            data = [None] * len(data_unordered)
            for (i, _) in enumerate(data):
                data[i] = data_unordered[
                    (i % self.num_classes) * (self.num_samples_per_class + 1)
                    + i // self.num_classes
                ]

            # Shuffle last N samples for the query set
            query_data = data[-self.num_classes :]
            random.shuffle(query_data)
            data[-self.num_classes :] = query_data

            imgs_batch = [
                image_file_to_array(img_path, self.dim_input) for _, img_path in data
            ]

            labels_batch = [label for label, _ in data]

            imgs[task_num] = np.array(imgs_batch).reshape(
                (self.num_samples_per_class + 1, self.num_classes, self.dim_input)
            )
            labels[task_num] = np.array(labels_batch).reshape(
                (self.num_samples_per_class + 1, self.num_classes, self.dim_output)
            )
        print(labels)

        return torch.from_numpy(imgs).to(self.device), torch.from_numpy(labels).to(
            self.device
        )
