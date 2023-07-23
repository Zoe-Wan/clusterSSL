from torchvision import transforms
from dataset.augmentation import GaussianBlur

def get_train_augment(dataset):
    size = 32
    if dataset == 'cifar10':
        
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    else:
        
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    normalize = transforms.Normalize(mean=mean, std=std)
    
    augs = [
        transforms.RandomResizedCrop(size=size, scale=(0.2, 1)),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        # transforms.RandomGrayscale(p=0.2),
    ]
    augs.extend([transforms.ToTensor(),normalize])
    train_transform = transforms.Compose(augs)
    return train_transform


def get_default_augment(dataset):
    size = 32
    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        
    normalize = transforms.Normalize(mean=mean, std=std)
    
    augs = [
        transforms.RandomResizedCrop(size=size, scale=(0.2, 1)),
        transforms.RandomHorizontalFlip(),
        
    ]
    augs.extend([transforms.ToTensor(),normalize])
    train_transform = transforms.Compose(augs)
    return train_transform



def get_test_augment(dataset):
    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    normalize = transforms.Normalize(mean=mean, std=std)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return test_transform


class MultiTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        imgs = []
        for t in self.transforms:
            imgs.append(t(x))
        return imgs
