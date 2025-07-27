import torch, torch.nn as nn, torch.optim as optim
from typing import Dict, Tuple, Union
from collections.abc import Mapping, Sequence
import torch


def train_encoder(
    model: torch.nn.Module,
    cutmix_criterion,
    loaders: Dict[str, torch.utils.data.DataLoader],
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    num_epochs: int = 200,
    logger=None,
):
    """Train encoder with CutMix and optional CSV logging.

    Args:
        model: The neural network to train.
        cutmix_criterion: Loss function that accepts CutMix targets.
        loaders: Dict containing "d0_train" and "d0_test" dataloaders.
        criterion: Standard criterion for validation.
        optimizer: Optimizer instance.
        scheduler: Learning‑rate scheduler.
        device: Computation device.
        num_epochs: Number of epochs (default 200).
        logger: CSVLogger instance for logging or ``None``.
    """

    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (imgs, targets) in enumerate(loaders["d0_train"]):
            imgs = imgs.to(device)

            # # Handle CutMix tuple targets
            if isinstance(targets, (list, tuple)):
                targets = tuple(
                    t.to(device) if isinstance(t, torch.Tensor) else t for t in targets
                )
                targets_for_acc = targets[0]  # original labels for accuracy
            else:
                targets = targets.to(device)
                targets_for_acc = targets

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = cutmix_criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets_for_acc.size(0)
            correct += predicted.eq(targets_for_acc).sum().item()

            # if batch_idx % 50 == 0:
            #     print(
            #         f"Epoch: {epoch + 1}/{num_epochs} | Batch: {batch_idx}/{len(loaders['d0_train'])} | "
            #         f"Loss: {loss.item():.3f} | Acc: {100. * correct / total:.2f}%"
            #     )

        train_acc = 100.0 * correct / total
        avg_train_loss = train_loss / len(loaders["d0_train"])

        # CSV log (train)
        if logger is not None:
            logger.log(
                stage="train",
                epoch=epoch + 1,
                acc=train_acc,
                loss=avg_train_loss,
                lr=scheduler.get_last_lr()[0],
            )

        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, targets in loaders["d0_test"]:
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100.0 * correct / total
        avg_test_loss = test_loss / len(loaders["d0_test"])

        print("\nEpoch {}/{} Summary:".format(epoch + 1, num_epochs))
        print(f"Train Loss: {avg_train_loss:.3f} | Test Loss: {avg_test_loss:.3f}")
        print(f"Test Accuracy: {acc:.2f}%")

        # CSV log (test)
        if logger is not None:
            logger.log(
                stage="test",
                epoch=epoch + 1,
                acc=acc,
                loss=avg_test_loss,
                lr=scheduler.get_last_lr()[0],
            )

        # Save best accuracy
        if acc > best_acc:
            best_acc = acc
            print(f"New best accuracy: {best_acc:.2f}%")
            if logger is not None:
                logger.log(
                    stage="best",
                    epoch=epoch + 1,
                    acc=best_acc,
                    loss=avg_test_loss,
                    lr=scheduler.get_last_lr()[0],
                )

        scheduler.step()
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}\n")

def setup_training_components(model):
    criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss (logits)
    optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return criterion, optimizer, scheduler, device

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
