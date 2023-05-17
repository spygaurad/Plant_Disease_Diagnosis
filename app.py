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
    image_file = np.array(image_file)
    output = model.infer_a_sample(image_file)
    print(output)
    return output


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)

