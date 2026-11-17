import torch
import torch.nn as nn


def train_model(
    model,
    train_loader,
    optimizer,
    device,
    epochs=30,
    use_amp=False
):

    criterion = nn.CrossEntropyLoss()
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device, non_blocking=amp_enabled)
            labels = labels.to(device, non_blocking=amp_enabled)

            # Limpa os gradientes anteriores
            optimizer.zero_grad()

            # Forward propagation (com Mixed Precision opcional em GPU)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Backpropagation e atualização dos pesos
            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # Acumula a perda
            running_loss += loss.item() * images.size(0)

            # Calcula as previsões
            predictions = outputs.argmax(dim=1)

            total += labels.size(0)

            correct += (
                predictions == labels
            ).sum().item()

        epoch_loss = running_loss / total
        epoch_accuracy = correct / total

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {epoch_loss:.4f} "
            f"Accuracy: {epoch_accuracy * 100:.2f}%"
        )

    return model