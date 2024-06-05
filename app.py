from flask import Flask, render_template, request
import torch
from torchvision.transforms import ToPILImage
import base64
import io
from PIL import Image
from vae import VAE

app = Flask(__name__)
model = VAE(attr_dim=40)  # Make sure this attribute dimension matches your model exactly
model.load_state_dict(torch.load('vae_with_attrs.pth', map_location=torch.device('cpu')))
model.eval()

@app.route('/', methods=['GET', 'POST'])
def index():
    # This list must match the order and exact number of your model's attributes
    default_attrs = [-1] * 40  # Assuming -1 for absent, 1 for present

    if request.method == 'POST':
        # Update attributes from the form, index carefully aligned with your attribute list
        for i, attr in enumerate([
            '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
            'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
            'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
            'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
            'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
            'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
            'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
            'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']):
            form_value = request.form.get(attr)
            # print(i, attr, form_value)
            default_attrs[i] = 1 if form_value == 'on' else -1
        img_str = generate_image(default_attrs)
        return render_template('index.html', image_data=img_str, attrs=default_attrs)
    else:
        return render_template('index.html', image_data=None, attrs=default_attrs)

def generate_image(attrs):
    z_dim = 20
    z = torch.randn(1, z_dim)
    attrs = torch.tensor([attrs]).float()
    with torch.no_grad():
        generated_image = model.decode(z, attrs)
    pil_image = ToPILImage()(generated_image.squeeze())

    # Calculate the new size as 3 to 4 times the original
    original_size = pil_image.size
    scale_factor = 3  # or 4, depending on how much you want to scale
    new_size = (original_size[0] * scale_factor, original_size[1] * scale_factor)
    pil_image = pil_image.resize(new_size, Image.LANCZOS)  # Use high-quality resampling

    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

if __name__ == '__main__':
    app.run(debug=True)
