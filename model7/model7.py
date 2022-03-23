import numpy as np
from PIL import Image, ImageOps, ImageEnhance

# ImageNet code should change this value
IMAGE_SIZE = 32


def int_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval .
  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
  return int(level * maxval / 10)


def float_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval.
  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
  return float(level) * maxval / 10.


def sample_level(n):
  return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _):
  return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
  return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
  level = int_parameter(sample_level(level), 4)
  return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
  degrees = int_parameter(sample_level(level), 30)
  if np.random.uniform() > 0.5:
    degrees = -degrees
  return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
  level = int_parameter(sample_level(level), 256)
  return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, level, 0, 0, 1, 0),
                           resample=Image.BILINEAR)


def shear_y(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, 0, level, 1, 0),
                           resample=Image.BILINEAR)


def translate_x(pil_img, level):
  level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, level, 0, 1, 0),
                           resample=Image.BILINEAR)


def translate_y(pil_img, level):
  level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, 0, 0, 1, level),
                           resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


augmentations = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y
]

augmentations_all = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y, color, contrast, brightness, sharpness
]

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

mixture_width = 3
mixture_depth = np.random.randint(1, 4)
aug_severity = 3
all_ops = 1
no_jsd = 0
num_workers = 4
epochs = 500
learning_rate = 0.1
momentum=0.9
decay=0.0005
print_freq = 10
class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 42

        self.conv1 = nn.Conv2d(3, 42, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(42)
        self.layer1 = self._make_layer(block, 42, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 84, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 168, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 336, num_blocks[3], stride=2)
        self.linear = nn.Linear(336, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def train(net, train_loader, optimizer, scheduler):
    """Train for one epoch."""
    net.train()
    loss_ema = 0.
    for i, (images, targets) in enumerate(train_loader):
        optimizer.zero_grad()

        if no_jsd:
            images = images.cuda()
            targets = targets.cuda()
            logits = net(images)
            loss = F.cross_entropy(logits, targets)
        else:
            images_all = torch.cat(images, 0).cuda()
            targets = targets.cuda()
            logits_all = net(images_all)
            logits_clean, logits_aug1, logits_aug2 = torch.split(
                logits_all, images[0].size(0))

            # Cross-entropy is only computed on clean images
            loss = F.cross_entropy(logits_clean, targets)

            p_clean, p_aug1, p_aug2 = F.softmax(
                logits_clean, dim=1), F.softmax(
                    logits_aug1, dim=1), F.softmax(
                        logits_aug2, dim=1)

            # Clamp mixture distribution to avoid exploding KL divergence
            p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
            loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_ema = loss_ema * 0.9 + float(loss) * 0.1
        # if i % print_freq == 0:
        #     print('Train Loss {:.3f}'.format(loss_ema))

    return loss_ema

def test(net, test_loader):
    """Evaluate network on given dataset."""
    net.eval()
    total_loss = 0.
    total_correct = 0
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.cuda(), targets.cuda()
            logits = net(images)
            loss = F.cross_entropy(logits, targets)
            pred = logits.data.max(1)[1]
            total_loss += float(loss.data)
            total_correct += pred.eq(targets.data).sum().item()

    return total_loss / len(test_loader.dataset), total_correct / len(
      test_loader.dataset)

def project1_model():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 +
                                             np.cos(step / total_steps * np.pi))

def aug(image, preprocess):
    """Perform AugMix augmentations and compute mixture.
    Args:
    image: PIL.Image input image
    preprocess: Preprocessing function which should return a torch tensor.
    Returns:
    mixed: Augmented and mixed image.
    """
    aug_list = augmentations
    if all_ops == 1:
        aug_list = augmentations_all

    ws = np.float32(np.random.dirichlet([1] * mixture_width))
    m = np.float32(np.random.beta(1, 1))

    mix = torch.zeros_like(preprocess(image))
    for i in range(mixture_width):
        image_aug = image.copy()
        depth = mixture_depth if mixture_depth > 0 else np.random.randint(
            1, 4)
        for _ in range(depth):
            op = np.random.choice(aug_list)
            image_aug = op(image_aug, aug_severity)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * preprocess(image_aug)

    mixed = (1 - m) * preprocess(image) + m * mix
    return mixed


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix augmentation."""

    def __init__(self, dataset, preprocess, no_jsd=False):
        self.dataset = dataset
        self.preprocess = preprocess
        self.no_jsd = no_jsd

    def __getitem__(self, i):
        x, y = self.dataset[i]
        if self.no_jsd:
            return aug(x, self.preprocess), y
        else:
            im_tuple = (self.preprocess(x), aug(x, self.preprocess),
                aug(x, self.preprocess))
            return im_tuple, y

    def __len__(self):
        return len(self.dataset)

import torchvision
import torchvision.transforms as transforms
#transform = transforms.Compose(
#    [transforms.AugMix(),
#    transforms.ToTensor(),
#    transforms.Normalize([0, 0, 0], [1, 1, 1])])

torch.manual_seed(1)
np.random.seed(1)
train_transform = transforms.Compose(
      [transforms.RandomHorizontalFlip(),
       transforms.RandomCrop(32, padding=4)])
preprocess = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize([0.5] * 3, [0.5] * 3)])

test_transform = preprocess

batch_size = 128


train_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)

train_data = AugMixDataset(train_data, preprocess, no_jsd)

train_loader = torch.utils.data.DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True)

test_data = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)

test_loader = torch.utils.data.DataLoader(
  test_data,
  batch_size=batch_size,
  shuffle=False,
  num_workers=num_workers,
  pin_memory=True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


import torch.optim as optim
net = project1_model()
criterion = nn.CrossEntropyLoss()
# optimizer = optim.RAdam(net.parameters(), lr=0.00015)

optimizer = torch.optim.SGD(
    net.parameters(),
    learning_rate,
    momentum=momentum,
    weight_decay=decay,
    nesterov=True)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
        step,
        epochs * len(train_loader),
        1,  # lr_lambda computes multiplicative factor
        1e-6 / learning_rate))
print(sum(p.numel() for p in net.parameters() if p.requires_grad))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
net.to(device)
print("Start Training")
correctness = []
bestcorrect = 0
for epoch in range(epochs):  # loop over the dataset multiple times

    train_loss_ema = train(net, train_loader, optimizer, scheduler)
    test_loss, test_acc = test(net, test_loader)
    is_best = test_acc > bestcorrect

    print(f'{epoch + 1} loss: {train_loss_ema / 2000:.3f}')
    if epoch % 10 == 9:
        test_loss, test_acc = test(net, test_loader)
        if test_acc > bestcorrect:
            best_path = f'./model9_{epoch + 1}_{test_acc}.pth'
            torch.save({
                'epoch':epoch,
                'model_state_dict':net.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'best_acc': test_acc}, best_path)
            bestcorrect = test_acc
        print(f'{epoch + 1} accuracy of the network on the 10000 test images: {test_acc} %')
        correctness.append(test_acc)

print('Finished Training')
PATH = './model9.pth'
torch.save(net.state_dict(), PATH)
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
print(correctness)

