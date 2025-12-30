from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import random_split

def get_data_loaders(batch_size):
    # transform_train = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),#随机水平翻转
    #     transforms.RandomCrop(96, padding=4),#随机裁剪
    #     transforms.ToTensor(),#将图片转化为张量
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])#归一化
    # ])
    transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomCrop(224, padding=16, padding_mode='reflect'), # 先裁剪（带 padding）
    transforms.RandomHorizontalFlip(p=0.5),                        # 再翻转
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.STL10(root='./data', split='train', download=True, transform=transform_train)
    test_dataset = datasets.STL10(root='./data', split='test', download=True, transform=transform_test)
#
    val_size = int(0.2 * len(train_dataset))  # 20%作为验证集
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
#
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)#

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader,  val_loader, test_loader