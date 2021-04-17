import wandb, torch
from dataloader import dataloader
from my_models import Predictor
import torchvision.models as models
# initalize run config
wandb.init(project="texture-descriptions",
    config={
        "approach": "Discriminative",
        "learning_rate": 0.0001,
        "architecture": "ResNet-101",
        "dataset": "DTD2",
        "device": "GTX1080",
        "epochs": 75,
        "batch_size": 50,
        "record_rate": 20,
    }
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # cuda device
# their resnet modifications are copied here
model = Predictor.Predictor(class_num=655, backbone='resnet101', pretrained_backbone=True, use_feats=(2,4,), fc_dims=(512,))
model.to(device)
criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config['learning_rate'])
wandb.watch(model, criterion=criterion) # send reports to wandb

train_data = dataloader.DTD2('dtd/images', 'image_descriptions.json', 'image_splits.json', split='train')
train_loader = torch.utils.data.DataLoader(train_data, batch_size=wandb.config['batch_size'], shuffle=True, num_workers=2)
epoch = 0
while epoch <= wandb.config['epochs']:
    # training
    step = 0
    for images, gts in train_loader:
        images = images.to(device)
        gts = gts.to(device)
        predicted = model(images)
        optimizer.zero_grad()
        loss = criterion(predicted, gts)
        loss.backward()
        optimizer.step()
        if step % wandb.config['record_rate'] == 0:
            wandb.log({"loss": loss})
    epoch += 1
    wandb.log({"epoch": epoch})
    print("Epoch:" + str(epoch - 1) + " finished!")
    torch.save(model.state_dict(), 'save_files/epoch' + str(epoch) + '.pth')
print("Done.")
