import os
import warnings

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter, Compose, RandomAffine, RandomHorizontalFlip, Resize, ToTensor

import attack.BackdoorBox as bb

from .exported_poisoned_dataset import ExportedPoisonedDataset
from .originalimagenet import Origdataset


def read_image(img_path, type=None):
    img = cv2.imread(img_path)
    if type is None:
        return img
    elif isinstance(type, str) and type.upper() == "RGB":
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(type, str) and type.upper() == "GRAY":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise NotImplementedError

def gen_grid(height, k):
    """Generate an identity grid with shape 1*height*height*2 and a noise grid with shape 1*height*height*2
    according to the input height ``height`` and the uniform grid size ``k``.
    """
    ins = torch.rand(1, 2, k, k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))  # a uniform grid
    noise_grid = nn.functional.upsample(ins, size=height, mode="bicubic", align_corners=True)
    noise_grid = noise_grid.permute(0, 2, 3, 1)  # 1*height*height*2
    array1d = torch.linspace(-1, 1, steps=height)  # 1D coordinate divided by height in [-1, 1]
    x, y = torch.meshgrid(array1d, array1d)  # 2D coordinates height*height
    identity_grid = torch.stack((y, x), 2)[None, ...]  # 1*height*height*2

    return identity_grid, noise_grid


class SecretDataset(Dataset):
    """Dataset used by ISSBA to train the steganography encoder."""

    def __init__(self, dataset, secret_size, seed):
        self.dataset = dataset
        rng = np.random.RandomState(seed)
        self.secrets = rng.binomial(1, 0.5, size=(len(dataset), secret_size)).astype(np.float32)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        return image, torch.from_numpy(self.secrets[index])


def _build_patch_pattern(img_height, img_width, channels=3):
    pattern = torch.zeros((channels, img_height, img_width), dtype=torch.uint8)
    weight = torch.zeros((channels, img_height, img_width), dtype=torch.float32)

    if img_height <= 128:
        pattern[:, -6:-3, -6:-3] = torch.randn((channels, 3, 3)) * 255
        weight[:, -6:-3, -6:-3] = 1.0
    else:
        print("Using the pattern size: 9*9")
        pattern[:, -12:-3, -12:-3] = torch.randn((channels, 9, 9)) * 255
        weight[:, -12:-3, -12:-3] = 1.0
    return pattern, weight


def _build_adaptive_patch_patterns(img_height, img_width, channels=3):
    patch_size = 5 if img_height <= 128 else 9
    margin = 3
    positions = [
        (margin, margin),
        (margin, img_width - margin - patch_size),
        (img_height - margin - patch_size, margin),
        (img_height - margin - patch_size, img_width - margin - patch_size),
    ]
    patterns = []
    generator = torch.Generator().manual_seed(1234)
    for top, left in positions:
        pattern = torch.zeros((channels, img_height, img_width), dtype=torch.uint8)
        pattern[:, top : top + patch_size, left : left + patch_size] = torch.randint(
            0,
            256,
            (channels, patch_size, patch_size),
            dtype=torch.uint8,
            generator=generator,
        )
        patterns.append(pattern)
    return patterns


def _load_optional_state_dict(model, ckpt_path):
    if not ckpt_path:
        return model
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    return model


def _dynamic_attack_placeholders(args, clean_train, clean_test, attack_name):
    if attack_name == "ISSBA" and args.issba_exported_poisoned_root:
        return (
            ExportedPoisonedDataset(os.path.join(args.issba_exported_poisoned_root, "train")),
            ExportedPoisonedDataset(os.path.join(args.issba_exported_poisoned_root, "val_pois")),
        )
    if args.testwpurified or args.purify_traindataset or args.purify_clean_test or args.purify_pois_test:
        raise NotImplementedError(
            f"{attack_name} generates poisoned datasets dynamically during attack training. "
            "Use --testwpurified false for attack training only, or set --issba_exported_poisoned_root to a "
            "streaming-exported ISSBA poisoned dataset."
        )
    warnings.warn(
        f"{attack_name} does not expose poisoned_train_dataset/poisoned_test_dataset before training; "
        "returning clean dataset placeholders for non-purification attack training."
    )
    return clean_train, clean_test

def process_dataset(args):

    trainset = None
    testset = None
    dataset_name = args.dataset.strip().lower()

    default_img_height = args.img_size # setting img height
    default_img_width = args.img_size # setting img width

    if dataset_name in {"cifar10", "cifar-10", "cifar"}:
        dataset = torchvision.datasets.CIFAR10
        transform_train = Compose([
            Resize((default_img_height, default_img_width)),
            ToTensor()
        ])
        trainset = dataset(args.datasets_root_dir, train=True, transform=transform_train, download=True)

        transform_test = Compose([
            Resize((default_img_height, default_img_width)),
            ToTensor()
        ])
        testset = dataset(args.datasets_root_dir, train=False, transform=transform_test, download=True)

        
    elif dataset_name == "gtsrb":
        
        
        transform_train = Compose([
            Resize((default_img_height, default_img_width)),
            ToTensor()
        ])

        trainset = Origdataset(args, args.datasets_root_dir,  split="train", transform=transform_train)

        transform_test = Compose([
            Resize((default_img_height, default_img_width)),
            ToTensor()
        ])

        testset = Origdataset(args, args.datasets_root_dir,  split="val",  transform=transform_test)
        
    else:
        
        transform_train = Compose([
            Resize((default_img_height, default_img_width)),
            ToTensor()
        ])

        trainset = Origdataset(args, args.datasets_root_dir,  split="train",  transform=transform_train)

        transform_test = Compose([
            Resize((default_img_height, default_img_width)),
            ToTensor()
        ])

        testset = Origdataset(args, args.datasets_root_dir,  split="val",  transform=transform_test)

    
    if args.attack_method == "BadNet":
        
        pattern, weight = _build_patch_pattern(default_img_height, default_img_width)

        backdoor_instance = bb.core.BadNets(
            train_dataset=trainset,
            test_dataset=testset,
            model=bb.core.models.ResNet(34, num_classes=args.classes),
            loss=nn.CrossEntropyLoss(),
            y_target=args.attack_target,
            poisoned_rate=args.poisoned_rate,
            pattern=pattern,
            weight=weight,
            seed=args.seed,
            deterministic=args.deterministic,
            poisoned_transform_train_index=1,
            poisoned_transform_test_index=1
        )
        
    elif args.attack_method == "PhysicalBA":
        
        pattern, weight = _build_patch_pattern(default_img_height, default_img_width)

        backdoor_instance = bb.core.PhysicalBA(
            train_dataset=trainset,
            test_dataset=testset,
            model=bb.core.models.ResNet(34, num_classes=args.classes),
            loss=nn.CrossEntropyLoss(),
            y_target=args.attack_target,
            poisoned_rate=args.poisoned_rate,
            pattern=pattern,
            weight=weight,
            seed=args.seed,
            deterministic=args.deterministic,
            poisoned_transform_train_index=1,
            poisoned_transform_test_index=1,
            physical_transformations = Compose([
            RandomHorizontalFlip(),
            ColorJitter(brightness=0.2,contrast=0.2), 
            RandomAffine(degrees=10,translate=(0.1, 0.1), scale=(0.8, 0.9))])
        )

    elif args.attack_method == "AdaptivePatch":
        backdoor_instance = bb.core.AdaptivePatch(
            train_dataset=trainset,
            test_dataset=testset,
            model=bb.core.models.ResNet(34, num_classes=args.classes),
            loss=nn.CrossEntropyLoss(),
            y_target=args.attack_target,
            poisoned_rate=args.poisoned_rate,
            covered_rate=args.adaptive_patch_covered_rate,
            patterns=_build_adaptive_patch_patterns(default_img_height, default_img_width),
            alphas=[1.0, 0.8, 0.6, 0.4],
            seed=args.seed,
            deterministic=args.deterministic,
            poisoned_transform_train_index=1,
            poisoned_transform_test_index=1,
        )

    elif args.attack_method == "Blended":
        pattern = torch.zeros((1, default_img_height, default_img_width), dtype=torch.uint8)
        pattern[0, :, :] = torch.randint(0, 255, size=(default_img_height, default_img_width))
        weight = torch.zeros((1, default_img_height, default_img_width), dtype=torch.float32)
        weight[0, :, :] = 0.2

        backdoor_instance = bb.core.Blended(
            train_dataset=trainset,
            test_dataset=testset,
            model=bb.core.models.ResNet(34, num_classes=args.classes),
            loss=nn.CrossEntropyLoss(),
            y_target=args.attack_target,
            poisoned_rate=args.poisoned_rate,
            pattern=pattern,
            weight=weight,
            seed=args.seed,
            deterministic=args.deterministic,
            poisoned_transform_train_index=1,
            poisoned_transform_test_index=1
        )
    
    elif args.attack_method == "WaNet":

        identity_grid, noise_grid = gen_grid(default_img_height, 256)
        backdoor_instance = bb.core.WaNet(
            train_dataset=trainset,
            test_dataset=testset,
            model=bb.core.models.ResNet(34, num_classes=args.classes),
            loss=nn.CrossEntropyLoss(),
            y_target=args.attack_target,
            poisoned_rate=args.poisoned_rate,
            identity_grid=identity_grid,
            noise_grid=noise_grid,
            noise=False,
            seed=args.seed,
            deterministic=args.deterministic,
            poisoned_transform_train_index=1,
            poisoned_transform_test_index=1
        )

    elif args.attack_method in {"LabelConsistent", "LC"}:
        pattern, weight = _build_patch_pattern(default_img_height, default_img_width)
        adv_model = bb.core.models.ResNet(34, num_classes=args.classes)
        if not args.lc_adv_model_ckpt:
            warnings.warn(
                "LabelConsistent is using a randomly initialized adversarial model. "
                "Set --lc_adv_model_ckpt for meaningful adversarial sample generation."
            )
        adv_model = _load_optional_state_dict(adv_model, args.lc_adv_model_ckpt)
        adv_dataset_dir = os.path.join(
            args.lc_adv_dataset_dir,
            f"{args.dataset}_target{args.attack_target}_rate{args.poisoned_rate}_seed{args.seed}",
        )

        backdoor_instance = bb.core.LabelConsistent(
            train_dataset=trainset,
            test_dataset=testset,
            model=bb.core.models.ResNet(34, num_classes=args.classes),
            adv_model=adv_model,
            adv_dataset_dir=adv_dataset_dir,
            loss=nn.CrossEntropyLoss(),
            y_target=args.attack_target,
            poisoned_rate=args.poisoned_rate,
            adv_transform=Compose([
                Resize((default_img_height, default_img_width)),
                ToTensor(),
            ]),
            steps=args.lc_pgd_steps,
            pattern=pattern,
            weight=weight,
            schedule={
                "device": "GPU" if torch.cuda.is_available() else "CPU",
                "CUDA_VISIBLE_DEVICES": str(args.gpulist[0]) if getattr(args, "gpulist", None) else "0",
                "GPU_num": 1,
                "batch_size": args.lc_adv_batch_size,
                "num_workers": args.lc_adv_num_workers,
            },
            seed=args.seed,
            deterministic=args.deterministic,
            poisoned_transform_train_index=1,
            poisoned_transform_test_index=1,
        )

    elif args.attack_method == "ISSBA":
        encoder_schedule = {
            "secret_size": args.issba_secret_size,
            "enc_height": default_img_height,
            "enc_width": default_img_width,
            "enc_in_channel": 3,
            "enc_total_epoch": 20,
            "enc_secret_only_epoch": 2,
            "enc_use_dis": False,
        }
        backdoor_instance = bb.core.ISSBA(
            dataset_name=dataset_name,
            train_dataset=trainset,
            test_dataset=testset,
            train_steg_set=SecretDataset(trainset, args.issba_secret_size, args.seed),
            model=bb.core.models.ResNet(34, num_classes=args.classes),
            loss=nn.CrossEntropyLoss(),
            y_target=args.attack_target,
            poisoned_rate=args.poisoned_rate,
            encoder_schedule=encoder_schedule,
            encoder=None,
            seed=args.seed,
            deterministic=args.deterministic,
        )
        poisoned_train_dataset, poisoned_test_dataset = _dynamic_attack_placeholders(
            args, trainset, testset, args.attack_method
        )
        return trainset, testset, poisoned_train_dataset, poisoned_test_dataset, backdoor_instance

    elif args.attack_method == "IAD":
        iad_dataset_name = dataset_name if dataset_name in {"cifar10", "mnist", "gtsrb"} else "gtsrb"
        backdoor_instance = bb.core.IAD(
            dataset_name=iad_dataset_name,
            train_dataset=trainset,
            test_dataset=testset,
            train_dataset1=trainset,
            test_dataset1=testset,
            model=bb.core.models.ResNet(34, num_classes=args.classes),
            loss=nn.CrossEntropyLoss(),
            y_target=args.attack_target,
            poisoned_rate=args.poisoned_rate,
            cross_rate=args.iad_cross_rate,
            lambda_div=args.iad_lambda_div,
            lambda_norm=args.iad_lambda_norm,
            mask_density=args.iad_mask_density,
            EPSILON=1e-7,
            seed=args.seed,
            deterministic=args.deterministic,
        )
        poisoned_train_dataset, poisoned_test_dataset = _dynamic_attack_placeholders(
            args, trainset, testset, args.attack_method
        )
        return trainset, testset, poisoned_train_dataset, poisoned_test_dataset, backdoor_instance

    else:
        raise NotImplementedError

    poisoned_train_dataset, poisoned_test_dataset = backdoor_instance.poisoned_train_dataset, backdoor_instance.poisoned_test_dataset

    return trainset, testset, poisoned_train_dataset, poisoned_test_dataset, backdoor_instance


def clean_ins(args, clean_train, clean_test):
    
        clean_instance = bb.core.Clean(
            train_dataset = clean_train,
            test_dataset = clean_test,
            model=bb.core.models.ResNet(34, num_classes=args.classes),
            loss=nn.CrossEntropyLoss(),
            schedule=None,
            seed=args.dataset_seed,
            deterministic=True)
        
        return clean_instance
    
def purified_ins(args, clean_train, clean_test, purified_train, purified_test):
    
        purified_instance = bb.core.Purified(
            train_dataset =clean_train,
            test_dataset= clean_test,
            model=bb.core.models.ResNet(34, num_classes=args.classes),
            loss=nn.CrossEntropyLoss(),
            schedule=None,
            seed=args.dataset_seed,
            deterministic=True,
            poisoned_train_dataset = purified_train,
            poisoned_test_dataset = purified_test)
        
        return  purified_instance
    
    
