from torch import nn, optim
from torchvision import transforms
from src.model import NeuralStylesTransfer
from src.utils import load_image
from src.criterion import criterion


transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


content = load_image('data/content/emrata.PNG', transform)
style = load_image('data/style/style.jpg', transform)

"""
the reason why i used the same content image as the output is that 
i want to see the result of the style transfer and saving time when trining 
"""
output = load_image('data/content/emrata.PNG', transform) 
output.requires_grad = True


model = NeuralStylesTransfer()
optimizer = optim.Adam([output], lr=0.05)

"""
I will use layer number 4, 8, 13, 20, 27 as the output
"""
content_features = model(content, layers=["4", "8"])
style_features = model(style, layers=["4", "8"])

if __name__ == '__main__':
    for i in range(1):
        output_features = model(output, layers = ['4', '8'])
        loss = criterion(content_features, style_features, output_features, output_features, style_weight=1e6)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())





