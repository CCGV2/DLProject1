import numpy as np
from PIL import Image, ImageOps, ImageEnhance

# ImageNet code should change this value
IMAGE_SIZE = 32


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
checkpoint = torch.load('zuzong.pth')


print(sum(p.numel() for p in net.parameters() if p.requires_grad))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
net.to(device)
net.load_state_dict(checkpoint)

test_loss, test_acc = test(net, test_loader)
print(f'Accuracy of the network on the 10000 test images: {100 * test_acc} %')


