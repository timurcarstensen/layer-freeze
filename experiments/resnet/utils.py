import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


def validate(
    model: torch.nn.Module,
    validloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Validate the model on the validation set."""
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        start = time.perf_counter()
        for data, target in validloader:
            data, target = data.to(device), target.to(device)

            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        end = time.perf_counter()
    return 1 - (correct / total), end - start


def create_model(num_classes: int = 10) -> nn.Module:
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def data_prep(
    batch_size: int,
    get_val_set: bool = True,
    dataloader_workers: int = 4,
    prefetch_factor: int | None = None,
) -> tuple:
    """Prepare CIFAR10 dataset for training and testing."""
    # Define dataset specific transforms and classes
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        ]
    )
    dataset_class = torchvision.datasets.CIFAR10
    num_classes = 10

    trainset = dataset_class(root="./data", train=True, download=True, transform=transform)
    testset = dataset_class(root="./data", train=False, download=True, transform=transform)

    if get_val_set:
        train_size = len(trainset) - 10000  # Reserve 10k samples for validation
        train_set, val_set = torch.utils.data.random_split(trainset, [train_size, 10000])
        validloader = torch.utils.data.DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=dataloader_workers,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=prefetch_factor,
        )
    else:
        train_set = trainset
        validloader = None

    trainloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=dataloader_workers,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=dataloader_workers,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
    )

    return trainloader, validloader, testloader, num_classes


def full_fidelity_training(
    epochs: int = 10,
    batch_size: int = 1024,
    learning_rate: float = 0.008,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    optimizer: str = "adam",
) -> dict:
    """Main training interface for HPO."""
    # Prepare data
    trainloader, validloader, _, num_classes = data_prep(batch_size=batch_size)

    # Define model with new parameters
    model = create_model(num_classes=num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    if optimizer.lower() == "adam":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            betas=[beta1, beta2],
            weight_decay=weight_decay,
        )
    elif optimizer.lower() == "sgd":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            momentum=beta1,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    if torch.cuda.is_available():
        model = model.cuda()

    # Training loop
    _start = time.time()
    forward_times = []
    backward_times = []
    model.train()
    for _ in range(epochs):
        running_loss = 0.0
        for data, target in trainloader:
            optimizer.zero_grad()

            # forward + backward + optimize
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            forward_start = time.time()
            outputs = model(data)
            forward_end = time.time()
            forward_times.append(forward_end - forward_start)

            loss = criterion(outputs, target)

            backward_start = time.time()
            loss.backward()
            backward_end = time.time()
            backward_times.append(backward_end - backward_start)

            optimizer.step()

            # print statistics
            running_loss += loss.item()
    _end = time.time()

    # Validation loop
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in validloader:
            images, labels = data
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    val_err = 1 - (correct / total)

    return {
        "val_err": val_err,
        "val_acc": val_accuracy,
        "cost": _end - _start,
    }
