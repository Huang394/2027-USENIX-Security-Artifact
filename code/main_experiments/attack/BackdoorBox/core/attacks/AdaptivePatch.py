"""
Adaptive patch backdoor attack.

This port keeps the original BackdoorBox lazy dataset behavior and adds support
for ConvIR-ZIP's Origdataset/ImageFolder-based datasets.
"""

import copy
import random

import torch
from PIL import Image
from torchvision.datasets import CIFAR10, DatasetFolder, MNIST
from torchvision.transforms import Compose
from torchvision.transforms import functional as F

from attack.originalimagenet import Origdataset
from settings import base_args

from .base import Base

args = base_args


class ModifyTarget:
    def __init__(self, y_target):
        self.y_target = y_target

    def __call__(self, y_target):
        return self.y_target


class AddDatasetFolderTrigger:
    def __init__(self, pattern, weight):
        if pattern is None:
            raise ValueError("Pattern can not be None.")
        if weight is None:
            raise ValueError("Weight can not be None.")

        self.pattern = pattern.unsqueeze(0) if pattern.dim() == 2 else pattern
        self.weight = weight.unsqueeze(0) if weight.dim() == 2 else weight
        self.res = self.weight * self.pattern
        self.weight = 1.0 - self.weight

    def _add_trigger(self, img):
        if img.dim() == 2:
            return (self.weight * img.unsqueeze(0) + self.res).squeeze().type(torch.uint8)
        return (self.weight * img + self.res).type(torch.uint8)

    def __call__(self, img):
        if isinstance(img, Image.Image):
            tensor_img = self._add_trigger(F.pil_to_tensor(img))
            if tensor_img.size(0) == 1:
                return Image.fromarray(tensor_img.squeeze().numpy(), mode="L")
            if tensor_img.size(0) == 3:
                return Image.fromarray(tensor_img.permute(1, 2, 0).numpy())
            raise ValueError("Unsupportable image shape.")
        if isinstance(img, torch.Tensor):
            return self._add_trigger(img)
        raise TypeError(f"img should be PIL.Image.Image or torch.Tensor. Got {type(img)}")


class PoisonedImageFolderDataset(DatasetFolder):
    def __init__(
        self,
        benign_dataset,
        y_target,
        poisoned_rate,
        covered_rate,
        patterns,
        alphas,
        poisoned_transform_index,
        poisoned_target_transform_index,
    ):
        super().__init__(
            benign_dataset.root,
            benign_dataset.loader,
            benign_dataset.extensions,
            benign_dataset.transform,
            benign_dataset.target_transform,
            None,
        )
        self._configure_poisoning(
            len(benign_dataset),
            y_target,
            poisoned_rate,
            covered_rate,
            patterns,
            alphas,
            poisoned_transform_index,
            poisoned_target_transform_index,
        )

    def _configure_poisoning(
        self,
        total_num,
        y_target,
        poisoned_rate,
        covered_rate,
        patterns,
        alphas,
        poisoned_transform_index,
        poisoned_target_transform_index,
    ):
        if not patterns:
            raise ValueError("patterns can not be empty.")
        if not alphas:
            raise ValueError("alphas can not be empty.")
        poisoned_num = int(total_num * poisoned_rate)
        covered_num = int(total_num * covered_rate)
        if poisoned_num < 0:
            raise ValueError("poisoned_num should be greater than or equal to zero.")
        if covered_num < 0:
            raise ValueError("covered_num should be greater than or equal to zero.")

        index_list = list(range(total_num))
        random.shuffle(index_list)
        self.poisoned_set = index_list[:poisoned_num]
        self.covered_set = index_list[poisoned_num : poisoned_num + covered_num]

        self.poisoned_transform = Compose([]) if self.transform is None else copy.deepcopy(self.transform)
        self.add_trigger_transforms = []
        for idx, pattern in enumerate(patterns):
            alpha = alphas[idx % len(alphas)]
            mask = torch.logical_or(torch.logical_or(pattern[0] > 0, pattern[1] > 0), pattern[2] > 0).float()
            self.add_trigger_transforms.append(AddDatasetFolderTrigger(pattern, mask * alpha))

        self.poisoned_transform_index = poisoned_transform_index
        self.poisoned_rate = poisoned_rate
        if poisoned_rate >= 1.0:
            for add_trigger_transform in self.add_trigger_transforms[: max(1, len(patterns) // 2)]:
                self.poisoned_transform.transforms.insert(self.poisoned_transform_index, add_trigger_transform)

        self.poisoned_target_transform = Compose([]) if self.target_transform is None else copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))

    def _poison_sample(self, sample, index):
        transform = copy.deepcopy(self.poisoned_transform)
        trigger_index = index % len(self.add_trigger_transforms)
        transform.transforms.insert(self.poisoned_transform_index, self.add_trigger_transforms[trigger_index])
        return transform(sample)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        poisoned_target = target

        if self.poisoned_rate >= 1.0:
            sample = self.poisoned_transform(sample)
            poisoned_target = self.poisoned_target_transform(target)
        elif index in self.poisoned_set:
            sample = self._poison_sample(sample, self.poisoned_set.index(index))
            poisoned_target = self.poisoned_target_transform(target)
        elif index in self.covered_set:
            sample = self._poison_sample(sample, self.covered_set.index(index))
        else:
            if self.transform is not None:
                sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, {"label_orig": target, "label_pois": poisoned_target}


class PoisonedDataset(PoisonedImageFolderDataset, Origdataset):
    def __init__(
        self,
        benign_dataset,
        y_target,
        poisoned_rate,
        covered_rate,
        patterns,
        alphas,
        poisoned_transform_index,
        poisoned_target_transform_index,
    ):
        Origdataset.__init__(
            self,
            args,
            args.datasets_root_dir,
            benign_dataset.split,
            transform=benign_dataset.transform,
            target_transform=benign_dataset.target_transform,
            download=True,
        )
        self._configure_poisoning(
            len(benign_dataset),
            y_target,
            poisoned_rate,
            covered_rate,
            patterns,
            alphas,
            poisoned_transform_index,
            poisoned_target_transform_index,
        )


class PoisonedMNIST(MNIST):
    def __init__(
        self,
        benign_dataset,
        y_target,
        poisoned_rate,
        pattern,
        alpha,
        poisoned_transform_index,
        poisoned_target_transform_index,
    ):
        super().__init__(
            benign_dataset.root,
            benign_dataset.train,
            benign_dataset.transform,
            benign_dataset.target_transform,
            download=True,
        )
        self.poisoned_set = frozenset(random.sample(range(len(benign_dataset)), int(len(benign_dataset) * poisoned_rate)))
        self.poisoned_transform = Compose([]) if self.transform is None else copy.deepcopy(self.transform)
        weight = (pattern > 0).float() * alpha
        self.poisoned_transform.transforms.insert(poisoned_transform_index, AddDatasetFolderTrigger(pattern, weight))
        self.poisoned_target_transform = Compose([]) if self.target_transform is None else copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode="L")
        if index in self.poisoned_set:
            img = self.poisoned_transform(img)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
        return img, target


class PoisonedCIFAR10(CIFAR10):
    def __init__(
        self,
        benign_dataset,
        y_target,
        poisoned_rate,
        pattern,
        alpha,
        poisoned_transform_index,
        poisoned_target_transform_index,
    ):
        CIFAR10.__init__(
            self,
            benign_dataset.root,
            benign_dataset.train,
            benign_dataset.transform,
            benign_dataset.target_transform,
            download=True,
        )
        self.poisoned_set = frozenset(random.sample(range(len(benign_dataset)), int(len(benign_dataset) * poisoned_rate)))
        self.poisoned_transform = Compose([]) if self.transform is None else copy.deepcopy(self.transform)
        weight = (pattern > 0).float() * alpha
        self.poisoned_transform.transforms.insert(poisoned_transform_index, AddDatasetFolderTrigger(pattern, weight))
        self.poisoned_target_transform = Compose([]) if self.target_transform is None else copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img)
        if index in self.poisoned_set:
            img = self.poisoned_transform(img)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
        return img, target


def CreatePoisonedDataset(
    benign_dataset,
    y_target,
    poisoned_rate,
    covered_rate,
    patterns,
    alphas,
    poisoned_transform_index,
    poisoned_target_transform_index,
):
    if isinstance(benign_dataset, Origdataset):
        return PoisonedDataset(
            benign_dataset,
            y_target,
            poisoned_rate,
            covered_rate,
            patterns,
            alphas,
            poisoned_transform_index,
            poisoned_target_transform_index,
        )
    if isinstance(benign_dataset, DatasetFolder):
        return PoisonedImageFolderDataset(
            benign_dataset,
            y_target,
            poisoned_rate,
            covered_rate,
            patterns,
            alphas,
            poisoned_transform_index,
            poisoned_target_transform_index,
        )
    if isinstance(benign_dataset, MNIST):
        return PoisonedMNIST(
            benign_dataset,
            y_target,
            poisoned_rate,
            patterns[0],
            alphas[0],
            poisoned_transform_index,
            poisoned_target_transform_index,
        )
    if isinstance(benign_dataset, CIFAR10):
        return PoisonedCIFAR10(
            benign_dataset,
            y_target,
            poisoned_rate,
            patterns[0],
            alphas[0],
            poisoned_transform_index,
            poisoned_target_transform_index,
        )
    raise NotImplementedError(f"Unsupported dataset type: {type(benign_dataset)}")


class AdaptivePatch(Base):
    def __init__(
        self,
        train_dataset,
        test_dataset,
        model,
        loss,
        y_target,
        poisoned_rate,
        covered_rate,
        patterns=None,
        alphas=None,
        poisoned_transform_train_index=1,
        poisoned_transform_test_index=1,
        poisoned_target_transform_index=0,
        schedule=None,
        seed=0,
        deterministic=False,
    ):
        super().__init__(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model=model,
            loss=loss,
            schedule=schedule,
            seed=seed,
            deterministic=deterministic,
        )
        if patterns is None:
            raise ValueError("AdaptivePatch requires patterns.")
        if alphas is None:
            raise ValueError("AdaptivePatch requires alphas.")

        self.poisoned_train_dataset = CreatePoisonedDataset(
            train_dataset,
            y_target,
            poisoned_rate,
            covered_rate,
            patterns,
            alphas,
            poisoned_transform_train_index,
            poisoned_target_transform_index,
        )
        self.poisoned_test_dataset = CreatePoisonedDataset(
            test_dataset,
            y_target,
            1.0,
            0.0,
            patterns,
            [1.0] * len(alphas),
            poisoned_transform_test_index,
            poisoned_target_transform_index,
        )
