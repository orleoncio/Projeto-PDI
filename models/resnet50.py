from torchvision.models import resnet50
import torch.nn as nn

def create_resnet50(num_classes=6):

    model = resnet50(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model