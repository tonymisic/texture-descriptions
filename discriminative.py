import wandb, torch
from dataloader import dataloader
from models import Predictor
import torchvision.models as models
# initalize run config
# wandb.init(project="texture-descriptions",
#     config={
#         "approach": "Discriminative",
#         "learning_rate": 0.001,
#         "architecture": "ResNet-101",
#         "dataset": "DTD2",
#         "device": "GTX1080"
#     }
# )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # cuda device
image_encoder = models.resnet101(pretrained=True)
image_encoder.to(device)
# wandb.watch(resnet101) # send reports to wandb

# training loop
train_data = dataloader.DTD2('dtd/images', 'image_descriptions.json', 'image_splits.json', split='train')
train_loader = torch.utils.data.DataLoader(train_data, batch_size=72, shuffle=False, num_workers=2)
for images, image_locations in train_loader:
    images.to(device)


