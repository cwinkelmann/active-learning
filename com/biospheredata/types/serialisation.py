from typing import Type, TypeVar
from pydantic import BaseModel
import json
from pathlib import Path

# Define a generic type for Pydantic models
T = TypeVar("T", bound=BaseModel)


def save_model_to_file(model: T, filepath: str) -> None:
    """
    Saves a Pydantic model to a JSON file.

    Args:
        model (T): The Pydantic model to save.
        filepath (str): The file path to save the model.
    """
    filepath = Path(filepath)
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(model.dict(), f, indent=4)


def load_model_from_file(model_class: Type[T], filepath: str) -> T:
    """
    Loads a Pydantic model from a JSON file.

    Args:
        model_class (Type[T]): The Pydantic model class.
        filepath (str): The file path to load the model from.

    Returns:
        T: The loaded Pydantic model instance.
    """
    filepath = Path(filepath)
    with filepath.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return model_class(**data)