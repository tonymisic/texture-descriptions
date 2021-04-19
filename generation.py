import wandb, torch, numpy as np
from dataloader import dataloader, utils
from my_models import Predictor, Metric
import torchvision.models as models
from torchtext.data.metrics import bleu_score
# initalize run config
wandb.init(project="texture-descriptions",
    config={
        "approach": "Discriminative",
        "architecture": "ResNet-101",
        "dataset": "DTD2",
        "device": "GTX1080",
        "batch_size": 50,
        "task": "Description-Generation"
    }
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # cuda device
# their resnet modifications are copied here
model = Predictor.Predictor(class_num=655)
model.load_state_dict(torch.load('save_files/discriminative/epoch76.pth'))
model.to(device)
model.eval()
wandb.watch(model) # send reports to wandb
eval_data = dataloader.DTD2gen('dtd/images', 'image_descriptions.json', 'image_splits.json', split='val')
eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=wandb.config['batch_size'], shuffle=False, num_workers=2)

predictions = []
scores = []
for images, gts in eval_loader:
    f_gt_captions, f_pred_captions = [], []
    images, gts = images.to(device), gts.to(device)
    for i in range(images.size()[0]):
        with torch.no_grad():
            temp = torch.unsqueeze(images[i].clone().detach(),  0)
            predicted = model(temp)
            _, best = torch.topk(predicted, 5)
            _, best_gt = torch.topk(torch.tensor(gts[i]), 5)
            pred_captions, gt_captions = [], []
            best, best_gt = best.flatten().tolist(), best_gt.tolist()
            for id in best:
                phrase = eval_data.data.phid_to_phrase(id).replace(" ", "")
                pred_captions.append(phrase)
            f_pred_captions.append(pred_captions)
            for id in best_gt:
                phrase = eval_data.data.phid_to_phrase(id).replace(" ", "")
                gt_captions.append(phrase)
            f_gt_captions.append(gt_captions)
    score = bleu_score(f_gt_captions, f_pred_captions)
    wandb.log({"score": score})
