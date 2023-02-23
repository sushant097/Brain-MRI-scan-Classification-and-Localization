import torchvision.models as models
import torch.nn as nn
import timm
import timm


def build_model(pretrained=True, fine_tune=True, num_classes=4):
    if pretrained:
        print("[INFO]: Loading pre-trained weights")
    else:
        print("[INFO]: NOt loading pre-Trained weights")
    model = timm.create_model("vit_base_patch32_224", pretrained=True, num_classes=num_classes)

    if fine_tune:
        print("[INFO]: Fine-tuning all layers...")
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False

    return model
