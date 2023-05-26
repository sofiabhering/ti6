from flask import Flask, request, jsonify, send_file
from flask_cors import CORS, cross_origin
import os
import sys

from face_classifier import run


app = Flask(__name__)
cors = CORS(app,support_credentials=True)

@app.route('/predict', methods=['POST'])
@cross_origin(supports_credentials=True)
def image_upload():
    files = request.files.getlist("images[]")

    print(f'files -> {files}', file=sys.stdout)

    processed_images = []  
    for file in files:
        filename = file.filename
        print(filename, file=sys.stdout)
        file.save(f'./uploads/{filename}')
        processed_images.append(f'./processed_{filename}')   
        run(f'./uploads/{filename}',filename)

    print(processed_images, file=sys.stdout)
    return jsonify({'processed_images': processed_images}), 200


if __name__ == '__main__':
    app.run()
