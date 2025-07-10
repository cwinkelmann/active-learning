import hydra
import omegaconf

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
import logging
from pathlib import Path

# Set up logging
log = logging.getLogger(__name__)


class SimpleCNN(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)


def create_dummy_data(batch_size, num_batches=100):
    """Create dummy MNIST-like data for demonstration"""
    for _ in range(num_batches):
        data = torch.randn(batch_size, 1, 28, 28)
        target = torch.randint(0, 10, (batch_size,))
        yield data, target


@hydra.main(version_base=None, config_path="config", config_name="config")
def train(cfg: DictConfig) -> float:
    # Set up logging
    log.info(f"Starting training with config:\n{OmegaConf.to_yaml(cfg)}")

    omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    # Initialize wandb with hydra config
    wandb.init(
        project=cfg.experiment.project,
        entity=cfg.experiment.entity,
        name=cfg.experiment.name,
        tags=cfg.experiment.tags,
        notes=cfg.experiment.notes,
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    # Set random seed
    torch.manual_seed(cfg.training.seed)

    # Create model
    model = SimpleCNN(dropout=cfg.model.dropout)

    # Create optimizer
    if cfg.model.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.model.learning_rate,
            weight_decay=cfg.model.weight_decay
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.model.learning_rate,
            weight_decay=cfg.model.weight_decay
        )

    # Loss function
    criterion = nn.NLLLoss()

    # Training loop
    best_loss = float('inf')

    for epoch in range(cfg.training.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        # Use dummy data for demonstration
        for batch_idx, (data, target) in enumerate(create_dummy_data(cfg.model.batch_size)):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

            if batch_idx % 20 == 0:
                log.info(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.6f}')

        # Calculate metrics
        avg_loss = total_loss / 100  # 100 batches
        accuracy = 100. * correct / total

        # Log to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_accuracy": accuracy,
            "learning_rate": cfg.model.learning_rate
        })

        # Update best loss
        if avg_loss < best_loss:
            best_loss = avg_loss

        log.info(f'Epoch {epoch}: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    # Log final metrics
    wandb.log({"best_loss": best_loss})

    # Save model artifact
    model_path = Path("model.pt")
    torch.save(model.state_dict(), model_path)

    artifact = wandb.Artifact("model", type="model")
    artifact.add_file(str(model_path))
    wandb.log_artifact(artifact)

    wandb.finish()

    log.info(f"Training completed. Best loss: {best_loss:.4f}")
    return best_loss


if __name__ == "__main__":
    train()