import wandb, torch, torch.nn as nn
from dataloader import dataloader
from my_models import Metric
import torchvision.models as models
# initalize run config
# wandb.init(project="texture-descriptions",
#     config={
#         "approach": "Metric-Learning",
#         "learning_rate": 0.0001,
#         "architecture": "ResNet-101",
#         "dataset": "DTD2",
#         "device": "GTX1080",
#         "epochs": 75,
#         "batch_size": 50,
#         "record_rate": 20,
#     }
# )

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # cuda device
# # their resnet modifications are copied here
# model = Predictor.Predictor(class_num=655, backbone='resnet101', pretrained_backbone=True, use_feats=(2,4,), fc_dims=(512,))
# model.to(device)
# criterion = nn.TripletMarginLoss(margin=1.0, p=2)
# optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config['learning_rate'])
# wandb.watch(model, criterion=criterion) # send reports to wandb

# train_data = dataloader.DTD2('dtd/images', 'image_descriptions.json', 'image_splits.json', split='train')
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=wandb.config['batch_size'], shuffle=True, num_workers=2)
# epoch = 0
# while epoch <= wandb.config['epochs']:
#     # training
#     step = 0
#     for images, gts in train_loader:
#         images = images.to(device)
#         gts = gts.to(device)
#         predicted = model(images)
#         optimizer.zero_grad()
#         loss = criterion(predicted, gts)
#         loss.backward()
#         optimizer.step()
#         if step % wandb.config['record_rate'] == 0:
#             wandb.log({"loss": loss})
#     epoch += 1
#     wandb.log({"epoch": epoch})
#     print("Epoch:" + str(epoch - 1) + " finished!")
#     torch.save(model.state_dict(), 'save_files/metric/epoch' + str(epoch) + '.pth')
# print("Done.")




device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # cuda device
# their resnet modifications are copied here
model = Metric.Metric()
model.to(device)
criterion_p = nn.TripletMarginLoss(margin=1.0, p=2) 
criterion_i = nn.TripletMarginLoss(margin=1.0, p=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_data = dataloader.DTD2Triple('dtd/images', 'image_descriptions.json', 'image_splits.json', split='train')
train_loader = torch.utils.data.DataLoader(train_data, batch_size=15, shuffle=True, num_workers=2)

for [anchor_image, anchor_embedding, pos_image, pos_embedding, neg_image, neg_embedding] in train_loader:
    anchor_image, anchor_embedding, pos_image = anchor_image.to(device), anchor_embedding.to(device), pos_image.to(device)
    pos_embedding, neg_image, neg_embedding = pos_embedding.to(device), neg_image.to(device), neg_embedding.to(device)
    anchor_features = model(anchor_image)
    pos_features = model(pos_image)
    neg_features = model(neg_image)
    optimizer.zero_grad()
    loss = criterion_p(anchor_embedding, pos_features, neg_features) + criterion_i(anchor_features, pos_embedding, neg_embedding)
    loss.backward()
    optimizer.step()
    break
    # record phrase, image, and overall loss
torch.save(model.state_dict(), 'save_files/metric/epoch' + str(0) + '.pth')
print("Done.")
