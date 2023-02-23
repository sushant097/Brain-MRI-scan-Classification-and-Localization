import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import gradio as gr

idx_to_class = {
    0:'glioma',
    1:'meningioma',
    2:'notumor',
    3:'pituitary'
}


def get_model(num_classes=4):
    model = models.efficientnet_b0(pretrained=False)
    # Change the final classification head.
    model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
    return model


def get_transform(IMAGE_SIZE=224):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.185, 0.185, 0.185],
            std=[0.204, 0.204, 0.204]
            )
    ])
    return transform


device = ('cuda' if torch.cuda.is_available() else "cpu")
transform = get_transform()
model = get_model()
checkpoint = torch.load("../models/efficientnet_model.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


def prediction(input_img):
    input_img = transform(input_img).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(input_img)
        prediction = F.softmax(prediction, dim=1)

    confidences, cat_ids = torch.topk(prediction, 4)
    outputs = {
        idx_to_class[idx.item()]: c.item() for c, idx in zip(confidences[0], cat_ids[0])
    }
    return outputs


gr.Interface(
    fn=prediction,
    inputs = gr.Image(type='pil', shape=(224, 224)),
    outputs = gr.Label(num_top_classes=5)
).launch(share=True)