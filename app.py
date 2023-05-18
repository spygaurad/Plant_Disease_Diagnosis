from flask import Flask, request
from Model import Model
import numpy as np 
from PIL import Image
from torchvision.transforms import transforms

app = Flask(__name__)
model = Model(trained=True)


@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['image']
    image_file = Image.open(image_file)

    # Convert the image to a tensor.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_file = transform(image_file).unsqueeze(0)
    output = model.infer_a_sample(image_file)
    print(output)
    return output


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5555)

