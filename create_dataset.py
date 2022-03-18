import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from torch.utils import data


class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))

# Dataloader 작성법
# 1. 데이터 파일 경로 설정
# 2. 이미지 전처리 (transform)
# 3. Dataset을 상속받아 나만의 데이터셋 인스턴스 생성 클래스 구현
# 4. Dataloader에 데이터셋 입력하여 사용
    
def create_dataset(args):
    """
    주어진 dataset을 load하여, dataloader를 return

    Args : dataset 종류, labeled target shot 수, feature extractor 종류
    Returns : source_loader, target_loader, target_loader_unl, target_loader_val, target_loader_test, class_list

    """
    base_path = './data/txt/{dataset}'.format(dataset=args.dataset) # 데이터셋 파일 목록 저장 위치
    root = './data/{dataset}/'.format(dataset=args.dataset)         # 데이터셋 위치

    image_set_file_s = os.path.join(base_path, 'labeled_source_images_' + args.source + '.txt')                 # Source 데이터 로드
    image_set_file_t = os.path.join(base_path, 'labeled_target_images_' + args.target + '_%d.txt' % (args.num)) # Labeled target 데이터 로드
    image_set_file_unl = os.path.join(base_path, 'unlabeled_target_images_' + args.target + '_%d.txt' % (args.num)) # unlabeled target 데이터 로드
    image_set_file_t_val = os.path.join(base_path, 'validation_target_images_' + args.target + '_3.txt')        # Validation target 데이터 로드

    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224

    # dataset transformation
    data_transforms = {
        'train': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    source_dataset = Imagelists(image_set_file_s, root=root, transform=data_transforms['train'])
    target_dataset = Imagelists(image_set_file_t, root=root, transform=data_transforms['val'])
    target_dataset_val = Imagelists(image_set_file_t_val, root=root, transform=data_transforms['val'])
    target_dataset_unl = Imagelists(image_set_file_unl, root=root, transform=data_transforms['val'])
    target_dataset_test = Imagelists(image_set_file_unl, root=root, transform=data_transforms['test'])

    class_list = return_classlist(image_set_file_s)

    print("%d classes in this dataset" % len(class_list))

    source_loader = \
        torch.utils.data.DataLoader(source_dataset,
                                    batch_size=args.bs, num_workers=3,
                                    shuffle=True, drop_last=True)
    target_loader = \
        torch.utils.data.DataLoader(target_dataset,
                                    batch_size=min(args.bs, len(target_dataset)), num_workers=3,
                                    shuffle=True, drop_last=True)
    target_loader_val = \
        torch.utils.data.DataLoader(target_dataset_val,
                                    batch_size=min(args.bs, len(target_dataset_val)), num_workers=3,
                                    shuffle=True, drop_last=True)
    target_loader_unl = \
        torch.utils.data.DataLoader(target_dataset_unl,
                                    batch_size=args.bs * 2, num_workers=3,
                                    shuffle=True, drop_last=True)
    target_loader_test = \
        torch.utils.data.DataLoader(target_dataset_test,
                                    batch_size=args.bs * 2, num_workers=3,
                                    shuffle=True, drop_last=True)

    return source_loader, target_loader, target_loader_unl, target_loader_val, target_loader_test, class_list


def create_dataset_test(args):
    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset
    image_set_file_s = os.path.join(base_path, args.source + '_all' + '.txt')
    image_set_file_test = os.path.join(base_path,
                                       'unlabeled_target_images_' +
                                       args.target + '_%d.txt' % (args.num))
    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224

    data_transforms = {
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    target_dataset_unl = Imagelists(image_set_file_test, root=root,
                                          transform=data_transforms['test'],
                                          test=True)
    class_list = return_classlist(image_set_file_s)

    print("%d classes in this dataset" % len(class_list))

    target_loader_unl = \
        torch.utils.data.DataLoader(target_dataset_unl,
                                    batch_size=args.bs * 2, num_workers=3,
                                    shuffle=False, drop_last=False)
    return target_loader_unl, class_list


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def make_dataset_fromlist(image_list):
    with open(image_list) as f:
        image_index = [x.split(' ')[0] for x in f.readlines()]
    with open(image_list) as f:
        label_list = []
        selected_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[1].strip()
            label_list.append(int(label))
            selected_list.append(ind)
        image_index = np.array(image_index)
        label_list = np.array(label_list)
    image_index = image_index[selected_list]
    return image_index, label_list


def return_classlist(image_list):
    with open(image_list) as f:
        label_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[0].split('/')[-2]
            if label not in label_list:
                label_list.append(str(label))
    return label_list


class Imagelists(data.Dataset):
    def __init__(self, image_list, root="./data/multi/", transform=None, target_transform=None, test=False):

        imgs, labels = make_dataset_fromlist(image_list)

        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = pil_loader
        self.root = root
        self.test = test

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is
            class_index of the target class.
        """

        path = os.path.join(self.root, self.imgs[index])
        target = self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.test:
            return img, target
        else:
            return img, target, self.imgs[index]

    def __len__(self):
        return len(self.imgs)
