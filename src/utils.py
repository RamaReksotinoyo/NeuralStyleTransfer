from PIL import Image
import torch
from torch import tensor as ndarray


def load_image(path: str, transform):
    """
    Load an image from a file path.
    Transformed to tensor and unsqueezed.
    """
    image = Image.open(path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image


def gram_matrix(tensor: ndarray) -> ndarray:
    """
    Calculate the gram matrix of a given tensor.
    """
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram


