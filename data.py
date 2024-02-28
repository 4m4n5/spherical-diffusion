import json
import os, random
import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image
import math
import blobfile as bf
import numpy as np

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

def resize_arr(pil_class, image_size, keep_aspect=True):
    pil_class = pil_class.resize(image_size, resample=Image.NEAREST)
    arr_class = np.array(pil_class)
    return arr_class

def get_circle_dot_data(args):
    train_transforms = torchvision.transforms.Compose([
        T.Resize((args.img_size,args.img_size)),  # args.img_size + 1/4 *args.img_size
        # T.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = CircleDotDataset(json_file=args.dataset_path, image_transform=train_transforms)

    return train_dataset

def get_nuclei_mask_data(args):
    train_transforms = torchvision.transforms.Compose([
        T.ToTensor(),
        # T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    all_masks = _list_image_files_recursively(os.path.join(args.dataset_path, 'classes', 'train' if args.is_train else 'test'))
    
    print("Len of Dataset:", len(all_masks))

    dataset = NucleiMaskDataset(mask_paths=all_masks, resolution=(args.img_size, args.img_size), is_train=args.is_train)

    return dataset


class NucleiMaskDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        mask_paths,
        resolution,
        shard=0,
        num_shards=1,
        random_crop=True,
        random_flip=True,
        is_train=True,
    ):
        super().__init__()
        self.is_train = is_train
        self.resolution = resolution
        self.local_masks = mask_paths[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_masks)

    def __getitem__(self, idx):
        out_dict = {}
        
        # Load mask
        path = self.local_masks[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_mask = Image.open(f)
            pil_mask.load()
        pil_mask = pil_mask.convert("L")

        # Resize mask
        arr_mask = resize_arr(pil_mask, self.resolution)

        # Random flip
        if self.random_flip and random.random() < 0.5:
            arr_mask = arr_mask[:, ::-1].copy()

        arr_mask = arr_mask[None, ]

        # Return Image
        return arr_mask


class CircleDotDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        json_file, 
        image_transform, 
        percentage=100.0,
    ):
        self.ann = json.load(open(json_file, "r"))

        # Prune dataset to percentage
        random.shuffle(self.ann)
        if percentage < 100.0:
            to_remove = int((100.0 - percentage) / 100.0 * len(self.ann))
            self.ann = self.ann[to_remove:]

        self.image_transform = image_transform

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, idx):
        item = self.ann[idx]

        if not os.path.isfile(item["image"]):
            raise FileNotFoundError(item["image"])

        image = Image.open(item["image"]).convert("RGB")
        image = self.image_transform(image)
        
        angle = item["angle"]
        clz = item["class"]

        return image, angle, int(clz)


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders  