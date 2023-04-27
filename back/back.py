from flask import Flask, request, send_file

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def image_upload():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    file.save(f'./uploads/{file.filename}')

    return send_file('./uploads/' + f'processed_{file.filename}', mimetype='image/jpeg'), 200


if __name__ == '__main__':
    app.run()
