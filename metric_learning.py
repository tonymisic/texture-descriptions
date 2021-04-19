import wandb, torch, torch.nn as nn
from dataloader import dataloader
from my_models import Metric
import torchvision.models as models
# initalize run config
wandb.init(project="texture-descriptions",
    config={
        "approach": "Metric-Learning",
        "learning_rate": 0.0001,
        "architecture": "ResNet-101",
        "Encoding": "BERT",
        "dataset": "DTD2",
        "device": "GTX1080",
        "epochs": 75,
        "batch_size": 15,
        "record_rate": 20,
        "triplet_margin": 1.0,
        "triplet_p": 2,
        "image_loss_weight": 1.0,
        "phrase_loss_weight": 1.0,
    }
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # cuda device
# their resnet modifications are copied here
model = Metric.Metric()
model.to(device)
criterion_p = nn.TripletMarginLoss(margin=wandb.config['triplet_margin'], p=wandb.config['triplet_p']) 
criterion_i = nn.TripletMarginLoss(margin=wandb.config['triplet_margin'], p=wandb.config['triplet_p']) 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
wandb.watch(model) # send reports to wandb
train_data = dataloader.DTD2Triple('dtd/images', 'image_descriptions.json', 'image_splits.json', split='train')
train_loader = torch.utils.data.DataLoader(train_data, batch_size=wandb.config['batch_size'], shuffle=True, num_workers=2)

epoch = 0
while epoch <= wandb.config['epochs']:
    # training
    step = 0
    for [anchor_image, anchor_embedding, pos_image, pos_embedding, neg_image, neg_embedding] in train_loader:
        anchor_image, anchor_embedding, pos_image = anchor_image.to(device), anchor_embedding.to(device), pos_image.to(device)
        pos_embedding, neg_image, neg_embedding = pos_embedding.to(device), neg_image.to(device), neg_embedding.to(device)
        anchor_features = model(anchor_image)
        pos_features = model(pos_image)
        neg_features = model(neg_image)
        optimizer.zero_grad()
        phrase_loss = criterion_p(anchor_embedding, pos_features, neg_features)
        image_loss = criterion_i(anchor_features, pos_embedding, neg_embedding)
        loss = wandb.config['phrase_loss_weight'] * phrase_loss + wandb.config['image_loss_weight'] * image_loss
        loss.backward()
        optimizer.step()
        if step % wandb.config['record_rate'] == 0:
            wandb.log({"overall_loss": loss, "phrase_loss": phrase_loss, "image_loss": image_loss})
    epoch += 1
    wandb.log({"epoch": epoch})
    print("Epoch:" + str(epoch - 1) + " finished!")
    torch.save(model.state_dict(), 'save_files/metric/epoch' + str(epoch) + '.pth')
print("Done.")