import torch
import torch.nn as nn


def train_model(
    model,
    train_loader,
    optimizer,
    device,
    epochs=30
):

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Limpa os gradientes anteriores
            optimizer.zero_grad()

            # Forward propagation
            outputs = model(images)

            # Calcula a perda
            loss = criterion(outputs, labels)

            # Backpropagation
            loss.backward()

            # Atualiza os pesos
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