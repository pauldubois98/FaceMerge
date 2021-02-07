import os
from flask import Flask, render_template, request, redirect, url_for, abort, send_from_directory
from werkzeug.utils import secure_filename
import merge
import imghdr

def validate_image(stream):
    header = stream.read(MAX_CONTENT_LENGTH)
    stream.seek(0)
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')


app = Flask(__name__)
UPLOAD_PATH = 'uploads'
MAX_CONTENT_LENGTH = 1024 * 1024
UPLOAD_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif']
OUTPUT_FILE =  os.path.join(UPLOAD_PATH, 'out.jpg')
OUTPUT_FILE =  'out.jpg'


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    images = []
    for uploaded_file in request.files.getlist('image_files'):
        filename = secure_filename(uploaded_file.filename)
        if filename != '':
            file_ext = os.path.splitext(filename)[1]
            if file_ext.lower() not in UPLOAD_EXTENSIONS:
                abort(400)
            if validate_image(uploaded_file.stream) not in UPLOAD_EXTENSIONS:
                abort(400)
            uploaded_file.save(os.path.join(UPLOAD_PATH, filename))
            output_filename = filename
        images.append( os.path.join(UPLOAD_PATH, filename) )
    
    merge.merge_images(images,  os.path.join(UPLOAD_PATH, OUTPUT_FILE) )
    for image in images:
        os.remove(image)

    return send_from_directory(UPLOAD_PATH, OUTPUT_FILE)
    return redirect(url_for('index'))


if __name__ == "__main__":
  app.run(host='0.0.0.0',debug=False)
