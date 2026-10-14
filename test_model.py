import torch

from models.resnet50 import create_resnet50
from optimizers.optimizer_factory import create_optimizer


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = create_resnet50(num_classes=6)

model = model.to(device)

optimizer = create_optimizer("adam", model)

print("Dispositivo:", device)
print("Modelo:", model.__class__.__name__)
print("Otimizador:", optimizer.__class__.__name__)

# Simula um lote com duas imagens RGB
images = torch.randn(2, 3, 224, 224).to(device)

model.eval()

with torch.no_grad():
    outputs = model(images)

print("Formato da saída:", outputs.shape)