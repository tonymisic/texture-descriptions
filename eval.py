import wandb, torch, numpy as np
from dataloader import dataloader, utils
from my_models import Predictor
import torchvision.models as models
# initalize run config
wandb.init(project="texture-descriptions",
    config={
        "approach": "Discriminative",
        "architecture": "ResNet-101",
        "dataset": "DTD2",
        "device": "GTX1080",
        "batch_size": 50
    }
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # cuda device
# their resnet modifications are copied here
model = Predictor.Predictor(class_num=655, backbone='resnet101', pretrained_backbone=True, use_feats=(2,4,), fc_dims=(512,))
model.load_state_dict(torch.load('save_files/discriminative/epoch76.pth'))
model.to(device)
model.eval()
wandb.watch(model) # send reports to wandb

eval_data = dataloader.DTD2('dtd/images', 'image_descriptions.json', 'image_splits.json', split='val')
eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=wandb.config['batch_size'], shuffle=False, num_workers=2)

predictions = []

for images, gts in eval_loader:
    images, gts = images.to(device), gts.to(device)
    with torch.no_grad():
        predicted = model(images)
    predictions.append(predicted.cpu())
predictions = np.asarray(predictions)
scores = np.vstack(predictions)
score1, score2 = utils.retrieve_eval(scores, mode='i2p'), utils.retrieve_eval(scores, mode='p2i')

# write results 
wandb.log({
    "Image2Phrase Mean Reciprocal Rank": score1['mean_reciprocal_rank'],
    "Image2Phrase R-Precision": score1['r_precision'],
    "Image2Phrase Mean Average Precision": score1['mean_average_precision'],
    
    "Phrase2Image Mean Reciprocal Rank": score2['mean_reciprocal_rank'],
    "Phrase2Image R-Precision": score2['r_precision'],
    "Phrase2Image Mean Average Precision": score2['mean_average_precision']
})
wandb.log({
    "Image2Phrase Precision@5": score1['precision_at_005'],
    "Image2Phrase Precision@20": score1['precision_at_020'],
    "Phrase2Image Precision@5": score2['precision_at_005'],
    "Phrase2Image Precision@20": score2['precision_at_020'],
    "Image2Phrase Recall@5": score1['recall_at_005'],
    "Image2Phrase Recall@20": score1['recall_at_020'],
    "Phrase2Image Recall@5": score2['recall_at_005'],
    "Phrase2Image Recall@20": score2['recall_at_020']
})