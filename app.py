import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import copy

def create_header():
    gr.Markdown("""
    # 🎨 Yapay Zeka Sanatçısı: Nöral Stil Transferi
    Bir **içerik fotoğrafı** ve bir **stil fotoğrafı** yükleyerek, yapay zekanın bu ikisini nasıl birleştirdiğini görün!
    """)

imsize = 256
loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor()])

def image_loader(image):
    image = loader(image).unsqueeze(0)
    return image.to(torch.device("cpu"), torch.float)

unloader = transforms.ToPILImage()

class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

cnn = models.vgg19(pretrained=True).features.eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)
    def forward(self, img):
        return (img - self.mean) / self.std

def get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img):
    normalization = Normalization(normalization_mean, normalization_std)
    content_losses = []
    style_losses = []
    model = nn.Sequential(normalization)
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        model.add_module(name, layer)
        if name in ['conv_4']:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)
        if name in ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i + 1)]
    return model, style_losses, content_losses

def run_style_transfer(cnn, normalization_mean, normalization_std, content_img, style_img, input_img, num_steps=300, style_weight=1000000, content_weight=1):
    model, style_losses, content_losses = get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img)
    input_img.requires_grad_(True)
    model.requires_grad_(False)
    optimizer = optim.LBFGS([input_img])
    run = [0]
    while run[0] <= num_steps:
        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0
            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss
            style_score *= style_weight
            content_score *= content_weight
            loss = style_score + content_score
            loss.backward()
            run[0] += 1
            return style_score + content_score
        optimizer.step(closure)
    with torch.no_grad():
        input_img.clamp_(0, 1)
    return input_img

def style_transfer_app(content_image, style_image):
    """This is the main function that Gradio will call."""
    content_img = image_loader(Image.fromarray(content_image))
    style_img = image_loader(Image.fromarray(style_image))
    input_img = content_img.clone()
    
    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std, content_img, style_img, input_img, num_steps=100)
    
    final_image = unloader(output.cpu().squeeze(0))
    return final_image

with gr.Blocks() as demo:
    create_header()
    with gr.Row():
        content_input = gr.Image(label="İçerik Fotoğrafı")
        style_input = gr.Image(label="Stil Fotoğrafı")
    
    submit_button = gr.Button("Sanat Eserini Yarat")
    
    output_image = gr.Image(label="Sonuç")
    
    submit_button.click(
        fn=style_transfer_app,
        inputs=[content_input, style_input],
        outputs=output_image
    )
    
    gr.Examples(
        examples=[["content.jpg", "style.jpg"]],
        inputs=[content_input, style_input]
    )

if __name__ == "__main__":
    demo.launch()