
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision import datasets
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

class ContrastiveCifar(Dataset):
    def __init__(self,mode='train',classes=10,aug=None,return_index=True):

        super(ContrastiveCifar, self).__init__()

        self.return_index = return_index
        if classes==10:                
            if mode=='train':
                if aug == None:
                    aug=get_train_augment('cifar10')
                self.data = datasets.CIFAR10(root='data', download=True)
            if mode=='test':
                if aug == None:
                    aug=get_test_augment('cifar10')
                self.data = datasets.CIFAR10(root='data', train=False, download=True)
        elif classes==100:                
            if mode=='train':
                if aug == None:
                    aug=get_train_augment('cifar100')
                self.data = datasets.CIFAR100(root='data', download=True)
            if mode=='test':
                if aug == None:
                    aug=get_test_augment('cifar100')
                self.data = datasets.CIFAR100(root='data', train=False, download=True)
        self.aug = aug
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, label = self.data[index]
        if self.return_index:
            return index, self.aug(img),self.aug(img), label
        return self.aug(img),self.aug(img), label


class Cifar(Dataset):
    def __init__(self,mode='train',classes=10,aug=None,return_index=True):

        super(Cifar, self).__init__()

        self.return_index = return_index
        if classes==10:                
            if mode=='train':
                if aug == None:
                    aug=get_train_augment('cifar10')
                self.data = datasets.CIFAR10(root='data', download=True)
            if mode=='test':
                if aug == None:
                    aug=get_test_augment('cifar10')
                self.data = datasets.CIFAR10(root='data', train=False, download=True)
        elif classes==100:                
            if mode=='train':
                if aug == None:
                    aug=get_train_augment('cifar100')
                self.data = datasets.CIFAR100(root='data', download=True)
            if mode=='test':
                if aug == None:
                    aug=get_test_augment('cifar100')
                self.data = datasets.CIFAR100(root='data', train=False, download=True)
        self.aug = aug
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, label = self.data[index]
        if self.return_index:
            return index, self.aug(img), label
        return self.aug(img), label
