import os
import time
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import timm
from timm.data import create_transform
from timm.loss import LabelSmoothingCrossEntropy
from timm.scheduler import CosineLRScheduler
from timm.utils import accuracy, AverageMeter
from PIL import Image
from tqdm import tqdm

# Set up argument parser
parser = argparse.ArgumentParser(description='Train a Vision Transformer model')
parser.add_argument('--train-dir', type=str, required=True, help='Path to training data')
parser.add_argument('--val-dir', type=str, required=True, help='Path to validation data')
parser.add_argument('--output-dir', type=str, default='./output/vit_model', help='Output directory')
parser.add_argument('--model', type=str, default='vit_base_patch16_224', help='Model architecture')
parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--weight-decay', type=float, default=0.05, help='Weight decay')
parser.add_argument('--img-size', type=int, default=224, help='Image size')
parser.add_argument('--num-classes', type=int, default=2, help='Number of classes')
parser.add_argument('--no-amp', action='store_true', help='Disable Automatic Mixed Precision')
parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--resume', type=str, default='', help='Resume from checkpoint')
parser.add_argument('--debug', action='store_true', help='Debug mode')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('vit-training')


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a black image if loading fails
            img = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            img = self.transform(img)

        return img, label


def load_from_directory(directory, valid_extensions=('.jpg', '.jpeg', '.png')):
    """Load images from directory with class subdirectories."""
    directory = Path(directory)

    if not directory.exists():
        raise ValueError(f"Directory not found: {directory}")

    # Get class folders
    classes = sorted([d.name for d in directory.iterdir()
                      if d.is_dir() and not d.name.startswith('.')])

    if not classes:
        raise ValueError(f"No class directories found in {directory}")

    logger.info(f"Found {len(classes)} classes: {', '.join(classes)}")

    # Create class to index mapping
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    # Collect images and labels
    image_paths = []
    labels = []

    for cls_name, idx in class_to_idx.items():
        cls_dir = directory / cls_name

        # Find all images in this class directory
        class_images = [p for p in cls_dir.glob('**/*')
                        if p.is_file() and p.suffix.lower() in valid_extensions
                        and not p.name.startswith('.')]

        if not class_images:
            logger.warning(f"No images found for class {cls_name}")
            continue

        # Add paths and labels
        image_paths.extend(class_images)
        labels.extend([idx] * len(class_images))

        logger.info(f"  - {cls_name}: {len(class_images)} images")

    return image_paths, labels, class_to_idx


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, amp_autocast, loss_scaler):
    """Train for one epoch."""
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()

    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]")

    for batch_idx, (images, targets) in enumerate(train_loader_tqdm):
        # Move data to device
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).long()  # Ensure targets are long

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass with AMP
        with amp_autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)

        # Backward pass with AMP
        if loss_scaler is not None:
            loss_scaler.scale(loss).backward()
            loss_scaler.step(optimizer)
            loss_scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Calculate accuracy
        acc1, acc5 = accuracy(outputs, targets, topk=(1, min(5, outputs.size(1))))

        # Update metrics
        batch_size = images.size(0)
        losses.update(loss.item(), batch_size)
        top1.update(acc1.item(), batch_size)

        # Update progress bar
        train_loader_tqdm.set_postfix({
            "loss": f"{losses.avg:.4f}",
            "acc@1": f"{top1.avg:.2f}%"
        })

    return losses.avg, top1.avg


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()

    val_loader_tqdm = tqdm(val_loader, desc="Validation")

    with torch.no_grad():
        for images, targets in val_loader_tqdm:
            # Move data to device
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).long()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)

            # Calculate accuracy
            acc1, acc5 = accuracy(outputs, targets, topk=(1, min(5, outputs.size(1))))

            # Update metrics
            batch_size = images.size(0)
            losses.update(loss.item(), batch_size)
            top1.update(acc1.item(), batch_size)

            # Update progress bar
            val_loader_tqdm.set_postfix({
                "loss": f"{losses.avg:.4f}",
                "acc@1": f"{top1.avg:.2f}%"
            })

    return losses.avg, top1.avg


def main():
    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging to file
    file_handler = logging.FileHandler(output_dir / 'training.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Log arguments
    logger.info(f"Arguments: {args}")

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else
                          'cpu')
    logger.info(f"Using device: {device}")

    # Create transforms
    train_transform = create_transform(
        input_size=(args.img_size, args.img_size),
        is_training=True,
        auto_augment='rand-m9-mstd0.5-inc1',
        interpolation='bicubic',
    )

    val_transform = create_transform(
        input_size=(args.img_size, args.img_size),
        is_training=False,
        interpolation='bicubic',
    )

    # Load datasets
    logger.info(f"Loading training data from {args.train_dir}")
    train_paths, train_labels, class_to_idx = load_from_directory(args.train_dir)

    logger.info(f"Loading validation data from {args.val_dir}")
    val_paths, val_labels, _ = load_from_directory(args.val_dir)

    # Create datasets and loaders
    train_dataset = ImageDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = ImageDataset(val_paths, val_labels, transform=val_transform)

    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # Create model
    logger.info(f"Creating model: {args.model}")
    model = timm.create_model(
        args.model,
        pretrained=True,
        num_classes=args.num_classes,
    )

    # Ensure all parameters require gradients
    for param in model.parameters():
        param.requires_grad = True

    logger.info(f"Model created: {args.model}")
    model = model.to(device)

    # Create loss function, optimizer, scheduler
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=args.epochs,
        lr_min=1e-6,
        warmup_lr_init=1e-6,
        warmup_t=3,
        cycle_limit=1,
        t_in_epochs=True,
    )

    # Set up AMP
    use_amp = not args.no_amp and device.type == 'cuda'
    amp_autocast = torch.cuda.amp.autocast if use_amp else lambda: torch.no_grad()
    loss_scaler = torch.cuda.amp.GradScaler() if use_amp else None

    logger.info(f"Using AMP: {use_amp}")

    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc = 0.0

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)

            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1

            if 'val_acc' in checkpoint:
                best_acc = checkpoint['val_acc']

            logger.info(f"Loaded checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            logger.warning(f"No checkpoint found at '{args.resume}'")

    # Training loop
    logger.info("Starting training")
    for epoch in range(start_epoch, args.epochs):
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, amp_autocast, loss_scaler
        )

        # Update learning rate
        scheduler.step(epoch + 1)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Print metrics
        logger.info(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'train_loss': train_loss,
        }

        torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch + 1}.pth")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(checkpoint, output_dir / "best_model.pth")
            logger.info(f"New best model saved with accuracy: {val_acc:.2f}%")

    logger.info(f"Training completed. Best accuracy: {best_acc:.2f}%")

    # Save final model
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': best_acc,
        },
        output_dir / "final_model.pth"
    )

    logger.info(f"Final model saved to {output_dir / 'final_model.pth'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.exception(f"Error in training: {e}")