import os

from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.models import resnet18
from torchvision.transforms import Compose, RandomCrop, ColorJitter, Resize, ToTensor, CenterCrop

from deepclustering3.config import ConfigManger
from deepclustering3.data.sampler import InfiniteRandomSampler
from demo.trainers import demoTrainer

cur_folder = os.path.abspath(os.path.dirname(__file__))
config_path = f"{cur_folder}/config.yaml"
config_parser = ConfigManger(base_path=config_path, )


def get_transforms():
    train_transform = Compose([
        Resize(32),
        RandomCrop(28),
        ColorJitter(brightness=[0.8, 1], contrast=[0.8, 1], saturation=[0.9, 1]),
        ToTensor()
    ])
    val_transform = Compose([
        Resize(32),
        CenterCrop(28),
        ToTensor()
    ])
    return train_transform, val_transform


with config_parser(scope="base") as config:
    train_transform, val_transform = get_transforms()
    train_set = MNIST(root="./data", download=True, train=True, transform=train_transform)
    val_set = MNIST(root="./data", download=True, train=False, transform=val_transform)
    train_loader = DataLoader(train_set, batch_size=16, sampler=InfiniteRandomSampler(train_set, shuffle=True),
                              num_workers=2)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=2)

    model = resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(512, 10)
    resume_from = config["Trainer"].pop("resume_from", None)
    trainer = demoTrainer(model=model, criterion=nn.CrossEntropyLoss(), tra_loader=iter(train_loader),
                          val_loader=val_loader, **config["Trainer"])

    from deepclustering3.epocher.hooks import EntropyMinHook

    hook = EntropyMinHook(weight=1, hook_name="entmin")

    trainer.register_hooks(hook)
    trainer.init(config=config)

    if resume_from:
        trainer.resume_from_path(resume_from)
    trainer.start_training()
