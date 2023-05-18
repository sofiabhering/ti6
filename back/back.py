from flask import Flask, request, send_file
from flask_cors import CORS, cross_origin

from face_classifier import run

app = Flask(__name__)
cors = CORS(app,support_credentials=True)

@app.route('/predict', methods=['POST'])
@cross_origin(supports_credentials=True)
def image_upload():
    if 'image' not in request.files:
        return 'No image uploaded', 400

    image = request.files['image']
    image.save(f'./images/{image.filename}')
    
    run(f'./uploads/{image.filename}')

    app.logger.info('uploading')
    # send_file('./uploads/' + f'processed_{image.filename}', mimetype='image/jpeg'),
    return {"age":21}, 200


if __name__ == '__main__':
    app.run()
