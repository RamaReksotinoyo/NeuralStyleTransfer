from torch.nn.functional import mse_loss
from src.utils import gram_matrix
import torch

def criterion(content_features: torch.Tensor, 
              style_features: torch.Tensor,
              output_contents, 
              output_styles, 
              content_weight: int = 1, 
              style_weight: float = 1e5
            ) -> torch.Tensor:
    """
    calculating the loss of the model
    """

    # Content Loss
    content_loss = 0
    for c, o in zip(content_features, output_contents):
        content_loss += mse_loss(c, o)
    
    # Style Loss
    style_loss = 0
    for s, o in zip(style_features, output_styles):
        style_texture = gram_matrix(s)
        output_texture = gram_matrix(o)
        style_loss += mse_loss(style_texture, output_texture)
        
    # Total loss
    loss = content_weight * content_loss + style_weight * style_loss
    
    return loss