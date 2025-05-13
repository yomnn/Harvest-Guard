
from flask import Flask, request, redirect, url_for, render_template, flash
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__ ,static_url_path='/static')
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Define class names for each model
model_class_names = {
    'mango': ['Mango_Cutting Weevil', 'Mango_healthy', 'Mango_Die Back', 'Mango_Gall Midge', 'Mango_Anthracnose', 'Mango_Bacterial Canker', 'Mango_Powdery Mildew', 'Mango_Sooty Mould'],
    'banana': ['banana_cordana', 'banana_healthy', 'banana_pestalotiopsis', 'banana_sigatoka'],
    'corn': ['Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy'],
    'cotton': ['Aphids', 'Army worm', 'Bacterial Blight', 'Healthy'],
    'eggplant': ['Eggplant_Healthy Leaf', 'Eggplant_Insect Pest Disease', 'Eggplant_Leaf Spot Disease', 'Eggplant_Mosaic Virus Disease', 'Eggplant_Small Leaf Disease', 'Eggplant White Mold Disease', 'Eggplant Wilt Disease'],
    'peach': ['Peach___Bacterial_spot', 'Peach___healthy'],
    'pepper': ['Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy'],
    'potato': ['Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight'],
    'rice': ['Bacterial leaf blight', 'Brown spot', 'Leaf smut'],
    'tomato': ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'],
    'grapes': ['Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy'],
    'apple': ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy'],
    'cherry': ['Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy'],
    'detect': ['mango', 'banana', 'corn','cotton','eggplant','peach','pepper','potato','rice','tomato','grapes'],

}

# Load your models (assuming they are TensorFlow/Keras models)
models = {
    'mango': tf.keras.models.load_model('models/mango_model.h5', compile=False),
    'banana': tf.keras.models.load_model('models/banana_model.h5', compile=False),
    'corn': tf.keras.models.load_model('models/corn_model.h5', compile=False),
    'cotton': tf.keras.models.load_model('models/cotton_model.h5', compile=False),
    'eggplant': tf.keras.models.load_model('models/eggplant_model.h5', compile=False),
    'peach': tf.keras.models.load_model('models/peach_model.h5', compile=False),
    'pepper': tf.keras.models.load_model('models/pepper_model.h5', compile=False),
    'potato': tf.keras.models.load_model('models/potato_model.h5', compile=False),
    'rice': tf.keras.models.load_model('models/rice_model.h5', compile=False),
    'tomato': tf.keras.models.load_model('models/tomato_model.h5', compile=False),
    'grapes': tf.keras.models.load_model('models/grapes_model.h5', compile=False),
    'cherry': tf.keras.models.load_model('models/cherry_model.h5', compile=False),
    'apple': tf.keras.models.load_model('models/Apple_model.h5', compile=False),
    'detect': tf.keras.models.load_model('models/detect_model.h5', compile=False),

}


for key, model in models.items():
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',  # Use string identifier for standard losses
        metrics=['accuracy']
    )

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/detect')
def detect():
    return render_template('detect.html')

@app.route('/TypePage')
def TypePage():
    return render_template('type.html')

@app.route('/contact')
def contact():
    return render_template('contact_us.html')

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/learn')
def learn():
    return render_template('learn.html')

@app.route('/apple-learn')
def apple():
    return render_template('apple-learn.html')
@app.route('/banana')
def banana():
    return render_template('banana.html')
@app.route('/cherry')
def cherry():
    return render_template('cherry.html')
@app.route('/corn')
def corn():
    return render_template('corn.html')
@app.route('/cotton')
def cotton():
    return render_template('cotton.html')
@app.route('/cucumber')
def cucumber():
    return render_template('cucumber.html')
@app.route('/eggplant')
def eggplant():
    return render_template('eggplant.html')
@app.route('/grape')
def grape():
    return render_template('grape.html')
@app.route('/mango')
def mango():
    return render_template('mango.html')
@app.route('/peach')
def peach():
    return render_template('peach.html')
@app.route('/pepper')
def pepper():
    return render_template('pepper.html')
@app.route('/potato')
def potato():
    return render_template('potato.html')
@app.route('/rice')
def rice():
    return render_template('rice.html')
@app.route('/soybean')
def soybean():
    return render_template('soybean.html')
@app.route('/strawberry')
def strawberry():
    return render_template('strawberry.html')
@app.route('/tomato-learn')
def tomato():
    return render_template('tomato-learn.html')

@app.route('/ttomato')
def ttomato():
    return render_template('ttomato.html')
@app.route('/tpotato')
def tpotato():
    return render_template('tpotato.html')
@app.route('/tcorn')
def tcorn():
    return render_template('tcorn.html')
@app.route('/tgrape')
def tgrape():
    return render_template('tgrape.html')
@app.route('/tcotton')
def tcotton():
    return render_template('tcotton.html')
@app.route('/tpepper')
def tpepper():
    return render_template('tpepper.html')
@app.route('/tpeach')
def tpeach():
    return render_template('tpeach.html')
@app.route('/tbanana')
def tbanana():
    return render_template('tbanana.html')
@app.route('/trice')
def trice():
    return render_template('trice.html')
@app.route('/tmango')
def tmango():
    return render_template('tmango.html')
@app.route('/teggplant')
def teggplant():
    return render_template('teggplant.html')
@app.route('/tapple')
def tapple():
    return render_template('tapple.html')
@app.route('/tcherry')
def tcherry():
    return render_template('tcherry.html')




@app.route('/TypePage', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            result = process_image_and_predict(filepath, request.form['model_type'])
            return render_template('result.html', result=result)
    return render_template('type.html')

def process_image_and_predict(filepath, model_type):
    image = Image.open(filepath).convert('RGB')
    image = image.resize((100, 100))  # Resize to the expected input size
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    model = models.get(model_type)
    if model:
        prediction = model.predict(image_array)
        predicted_index = np.argmax(prediction)
        class_names = model_class_names.get(model_type, [])
        if class_names:
            predicted_class = class_names[predicted_index]
            return predicted_class
        else:
            return "Class names not defined for this model"
    else:
        flash('Model not found')
        return 'Error: Model not found'

if __name__ == '__main__':
    app.run(debug=True)
