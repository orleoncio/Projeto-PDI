import copy
import torch
import torch.nn as nn


@torch.no_grad()
def _evaluate_accuracy(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()
    return correct / total if total > 0 else 0.0


def train_model(
    model,
    train_loader,
    optimizer,
    device,
    epochs=30,
    use_amp=False,
    scheduler=None,
    val_loader=None,
    max_grad_norm=None,
    return_training_info=False,
    validation_label="Val",
):

    criterion = nn.CrossEntropyLoss()
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_val_acc = -1.0
    best_epoch = None
    best_model_wts = None

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

            # Backpropagation com Gradient Clipping e atualização dos pesos
            if amp_enabled:
                scaler.scale(loss).backward()
                if max_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            # Acumula a perda
            running_loss += loss.item() * images.size(0)

            # Calcula as previsões
            predictions = outputs.argmax(dim=1)

            total += labels.size(0)

            correct += (
                predictions == labels
            ).sum().item()

        if scheduler is not None:
            scheduler.step()

        epoch_loss = running_loss / total
        epoch_accuracy = correct / total

        if val_loader is not None:
            val_acc = _evaluate_accuracy(model, val_loader, device)
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                best_model_wts = copy.deepcopy(model.state_dict())
            star = " ★ [BEST]" if is_best else ""
            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Loss: {epoch_loss:.4f} "
                f"Train Acc: {epoch_accuracy * 100:.2f}% | "
                f"{validation_label} Acc: {val_acc * 100:.2f}%{star}",
                flush=True
            )
        else:
            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Loss: {epoch_loss:.4f} "
                f"Accuracy: {epoch_accuracy * 100:.2f}%",
                flush=True
            )

    if best_model_wts is not None:
        print(f"Restaurando o melhor checkpoint de validação (Val Acc: {best_val_acc * 100:.2f}%)")
        model.load_state_dict(best_model_wts)

    if return_training_info:
        return model, {"best_val_accuracy": best_val_acc, "best_epoch": best_epoch}
    return model
