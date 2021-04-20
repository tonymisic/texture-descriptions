import wandb, torch, numpy as np
from dataloader import dataloader, utils
from dataloader import gradcam
from my_models import Predictor, Metric
import torchvision.models as models
from PIL import Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # cuda device
# their resnet modifications are copied here
model = Predictor.Predictor(class_num=655)
model.load_state_dict(torch.load('save_files/discriminative/epoch76.pth'))
model.to(device)
model.eval()
eval_data = dataloader.DTD2('dtd/images', 'image_descriptions.json', 'image_splits.json', split='val')
eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=5, shuffle=True, num_workers=2)

grad = gradcam.ScoreCam(model, 4)
maximum_images = []
cam = None
for images, gts in eval_loader:
    images = images.to(device)
    cam = grad.generate_cam(images, target_class=0)
    maximum_images.append(cam)
    img = Image.fromarray(cam, 'RGB')
    img.save('test.png')
    break
