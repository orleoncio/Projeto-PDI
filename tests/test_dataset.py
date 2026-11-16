from data.hep2_dataset import create_dataloaders


train_loader, test_loader = create_dataloaders()

print("Imagens de treino:", len(train_loader.dataset))
print("Imagens de teste:", len(test_loader.dataset))

images, labels = next(iter(train_loader))

print("Formato das imagens:", images.shape)
print("Formato dos labels:", labels.shape)
print("Classes do primeiro lote:", labels)