import json
import ntpath
from Config import *
from Main import predict
from utils.utils import file2zip, clear

from flask import Flask, render_template, request
from flask_uploads import UploadSet, IMAGES, configure_uploads, patch_request_class

app = Flask(__name__, template_folder='front-end', static_folder='front-end/Static_Files')


@app.route('/', methods=["GET"])
def index():
    return render_template('index.html')


upload_path = os.path.join(ROOT_PATH, 'front-end', 'Static_Files', 'uploads')
download_path = os.path.join(ROOT_PATH, 'front-end', 'Static_Files', 'downloads')
clear(upload_path, download_path)

app.config['UPLOADED_PHOTOS_DEST'] = upload_path
images = UploadSet('photos', IMAGES)
configure_uploads(app, images)
patch_request_class(app)


@app.route('/upload_files', methods=['GET', 'POST'])
def upload():
    clear(upload_path, download_path)
    user_ip = request.remote_addr
    file_paths = []
    file_urls = []

    save_path = os.path.join(upload_path, user_ip)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    global images
    app.config['UPLOADED_PHOTOS_DEST'] = save_path
    configure_uploads(app, images)

    file_list = request.files.getlist('file')
    for file in file_list:
        filename = images.save(file)
        file_urls.append(images.url(filename))
        file_paths.append(os.path.join(save_path, filename))
    data = {'user_ip': user_ip, 'file_url': file_urls, 'file_paths': file_paths}
    return json.dumps(data)


@app.route('/start_predict', methods=['GET', 'POST'])
def start_predict():
    file_paths = request.get_json()['file_paths']
    user_ip = request.remote_addr
    path = '/Static_Files/downloads/' + user_ip + '/'
    save_path = os.path.join(download_path, user_ip)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    generated_img_list = []
    try:
        predict(file_paths, user_ip, isWeb=True)
    except RuntimeError:
        Error = "当前有其他用户正在使用，请稍后再试！"
        data = {'Status': 'Error', 'Message': Error, 'file_url': []}
        return json.dumps(data)
    for file_path in file_paths:
        full_file_name = ntpath.basename(file_path)
        file_name = os.path.splitext(full_file_name)[0] + '_fake_B_2deploy.png'
        generated_image_url = os.path.join(path, file_name)
        generated_img_list.append(generated_image_url)
    file2zip(file_paths, save_path)
    data = {'user_ip': user_ip, 'Status': "Work", 'Message': "", 'file_url': generated_img_list}
    return json.dumps(data)


# @app.route('/download_images', methods=['GET', 'POST'])
# def download():
#     memory_file = BytesIO()
#     with zipfile.ZipFile(memory_file, 'w') as zf:
#         download_list = []
#         for file_path in file_paths:
#             full_file_name = ntpath.basename(file_path)
#             file_name = os.path.splitext(full_file_name)[0] + '_fake_B.png'
#             generated_image = os.path.join(save_dir, file_name)
#             download_list.append(generated_image)
# The model:
# ----------------------------------------------------------------------------------------
# memory_file = BytesIO()
# with zipfile.ZipFile(memory_file, 'w') as zf:
#     files = result['files']
#     for individualFile in files:
#         data = zipfile.ZipInfo(individualFile['fileName'])
#         data.date_time = time.localtime(time.time())[:6]
#         data.compress_type = zipfile.ZIP_DEFLATED
#         zf.writestr(data, individualFile['fileData'])
# memory_file.seek(0)
# return send_file(memory_file, attachment_filename='capsule.zip', as_attachment=True)
# ----------------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8531)

# todo: fix up the bug:
#  When user secondly upload data the same to the first,
#  uploaded images won't be deployed.
