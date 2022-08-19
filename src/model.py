import torch
from torch import nn
from torchvision.models import vgg16


class NeuralStylesTransfer(nn.Module):
    """
    This is my nst model, the reason why i used base model vgg
     because the first paper of nst 'A Neural Algorithm of Artistic Style' by Gatys et al.
    """
    def __init__(self) -> None:
        super().__init__()
        model = vgg16(pretrained=True).eval()
        self.model = model.features
        self.freeze()

    def forward(self, x: torch.Tensor, layers: torch.Tensor) -> torch.Tensor:
        features = []
        for i, layer in self.model._modules.items():
            x = layer(x)
            if i in layers:
                features.append(x)
        return features

    def freeze(self) -> None:
        for param in self.model.parameters():
           param.requires_grad = False