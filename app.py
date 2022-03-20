from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import argparse


app = Flask(__name__)
model = tf.keras.models.load_model('model')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    classes = ['chair', 'clock', 'telephone']
    try:
        image = request.files.get('image').read()
        image = tf.image.decode_image(image, expand_animations=False)
        image = tf.keras.applications.efficientnet_v2.preprocess_input(image)
        image = tf.image.resize(image, (128, 128))
        image = tf.expand_dims(image, axis=0)

        if image.shape[-1] != 3:
            return jsonify({"error": "Number of image channels must be equal to 3."}), 400

        preds = model.predict(image)
        preds = tf.Variable(preds)
        probas = tf.keras.activations.softmax(preds).numpy().reshape(-1)
        prediction = np.argmax(probas)
        probas = np.round_(probas * 100, decimals=2)
        result = {
            'prediction': classes[prediction],
            'probas': {str(class_): str(proba) for class_ , proba in zip(classes, probas)}
        }

        return jsonify(result), 200

    except Exception as ex:
        result = {'errer': 'Input must be an image.'}
        return jsonify(result), 400


@app.route('/statistic', methods=['GET'])
def statistic():   
    result = {'confusion_matrix': {}} 
    classes = ['chair', 'clock', 'telephone']
    precisions = [.97, 1, .98]    
    recalls = [1, .98, .98]
    conf_mat = np.array([[223, 0, 1],[2, 219, 3], [5, 0, 219]])

    for class_, precision, recall, vector in zip(classes, precisions, recalls, conf_mat):
        result[class_] = {'precision': str(precision), 'recall': str(recall)}
        result['confusion_matrix'][class_] = str(vector)

    return jsonify(result), 200
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', default=5000, type=int, help='port number') 

    args = parser.parse_args()
    app.run(port=args.port)
