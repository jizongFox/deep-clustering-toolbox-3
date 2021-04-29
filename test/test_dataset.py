import shutil
from unittest import TestCase

import torch

from deepclustering3.augment import SequentialWrapper, SequentialWrapperTwice
from deepclustering3.augment.pil_augment import ToLabel
from deepclustering3.data.dataset.acdc import ACDCDataset
from deepclustering3.data.dataset.base import extract_sub_dataset_based_on_scan_names


class TestACDCDownload(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._root = "./data"
        try:
            shutil.rmtree(self._root)
        except FileNotFoundError:
            pass

    def test_downloading_acdc(self):
        dataset = ACDCDataset(root_dir=self._root, mode="train", transforms=None)

    def tearDown(self) -> None:
        super().tearDown()


class TestACDCTransform(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._root = "./data"

    def test_single_transform(self):
        from torchvision.transforms import Compose, RandomCrop, RandomRotation, ColorJitter, ToTensor, RandomResizedCrop
        transforms = SequentialWrapper(
            com_transform=Compose([RandomRotation(45), RandomCrop(224), RandomResizedCrop(size=192, scale=(0.8, 1.2))]),
            image_transform=Compose(
                [ColorJitter(brightness=[0.8, 1.2], contrast=[0.8, 1.2], saturation=1), ToTensor()]),
            target_transform=ToLabel()
        )

        dataset = ACDCDataset(root_dir=self._root, mode="train", transforms=transforms, )
        (image, target), filename = dataset[4]
        from deepclustering3.viewer import multi_slice_viewer_debug
        import matplotlib.pyplot as plt
        multi_slice_viewer_debug(image, target, no_contour=True)
        plt.show()

    def test_twice_transform(self):
        from torchvision.transforms import Compose, RandomCrop, RandomRotation, ColorJitter, ToTensor
        transforms = SequentialWrapperTwice(
            com_transform=Compose([RandomRotation(45), RandomCrop(224)], ),
            image_transform=Compose(
                [ColorJitter(brightness=[0.8, 1.2], contrast=[0.8, 1.2], saturation=1), ToTensor()]),
            target_transform=ToLabel(),
            total_freedom=False
        )

        dataset = ACDCDataset(root_dir=self._root, mode="train", transforms=transforms, )
        (image1, image2, target1, target2), filename = dataset[4]
        from deepclustering3.viewer import multi_slice_viewer_debug
        import matplotlib.pyplot as plt
        multi_slice_viewer_debug(torch.cat([image1, image2], dim=0), torch.cat([target1, target2], dim=0),
                                 no_contour=True)
        plt.show()

    def test_scans(self):
        tra_set = ACDCDataset(root_dir=self._root, mode="train", )
        test_set = ACDCDataset(root_dir=self._root, mode="val", )
        train_scans = set(tra_set.get_scan_list())
        assert len(train_scans) == 175

        test_scans = set(test_set.get_scan_list())
        assert len(test_scans) == 25

        labeled_scans = sorted(train_scans)[:10]
        unlabeled_scans = sorted(train_scans)[10:]

        labeled_set = extract_sub_dataset_based_on_scan_names(tra_set, labeled_scans, )
        unlabeled_set = extract_sub_dataset_based_on_scan_names(tra_set, unlabeled_scans, )
        del train_scans
        assert len(labeled_set.get_scan_list()) == 10
        assert len(unlabeled_set.get_scan_list()) == 165
