import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from base import BaseModel


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super(MnistModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class MyModel(BaseModel):
    def __init__(self, num_classes=3):
        super(MyModel, self).__init__()
        # Get pretrained VGG16 model
        vgg16_model = models.vgg16(pretrained=True)
        # Get features from VGG16
        self.features = vgg16_model.features
        # Remove the last two layers (fc and activation)
        self.classifier = nn.Sequential(
            # TODO: How to load target from CVS with 3 classes instead of 4?
            nn.Linear(61440, 4)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.sigmoid(x)
