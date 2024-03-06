from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)
model = load_model('./food_classification_model.h5')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the uploaded image
        file = request.files['file']
        img = Image.open(file.stream)
        img = img.resize((224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # Make a prediction
        pred = model.predict(img)
        class_id = np.argmax(pred)
        class_name = get_class_name(class_id)

        # Render the prediction result
        return render_template('result.html', class_name=class_name)

    # Render the upload form
    return render_template('index.html')


def get_class_name(class_id):
    class_names = {
        0: 'ven pongal',
        1: 'vada pav',
        2: 'upma',
        3: 'tandoori chicken',
        4: 'samosa',
        5: 'poori',
        6: 'paniyaram',
        7: 'noodles',
        8: 'meduvadai',
        9: 'kathi roll',
        10: 'idly',
        11: 'halwa',
        12: 'gulab jamun',
        13: 'dosa',
        14: 'dhokla',
        15: 'chappati',
        16: 'chaat',
        17: 'butternaan',
        18: 'bisebelebath',
        19: 'biriyani'
    }
    return class_names.get(class_id, "Unknown")

if __name__ == '__main__':
    app.run(port=8000,debug=True)
