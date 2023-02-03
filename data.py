import json
import os, random
import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image


def get_data(args):
    train_transforms = torchvision.transforms.Compose([
        T.Resize((args.img_size,args.img_size)),  # args.img_size + 1/4 *args.img_size
        # T.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if args.data_type == "json":
        train_dataset = CircleDotDataset(json_file=args.dataset_path, image_transform=train_transforms)
    if args.data_type == "image_folder":
        train_dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=train_transforms)

    return train_dataset


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
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders  