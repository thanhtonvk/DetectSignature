from model import get_model
from preprocessing import img_preprocess
import os
import faiss
from detect import get_mask
import cv2
model = get_model()
flower_index = faiss.IndexFlatL2(128)
list_object = []
root_folder = 'static/sign/'
list_dir = os.listdir(root_folder)



k = 3

for idx,f in enumerate(list_dir):
  image = get_mask(root_folder+f)
  img = img_preprocess(image, expand=True)


  embedded = model.predict(img,verbose=False)  


  flower_index.add(embedded)
  obj = {'name':f.split('.')[0], 'sign':image }
  list_object.append(obj)


def predict(path):
    original =  get_mask(path)
    img_prep = img_preprocess(original, expand=True)
    test_fea = model.predict(img_prep) 
    f_dists, f_ids = flower_index.search(test_fea, k=k)
    return f_dists, f_ids


def get_object(f_dists, f_ids):
  list_near = []
  for i in range(k):
    obj = {'distance':f_dists[0][i],'id': f_ids[0][i], 'path':list_dir[f_ids[0][i]]}
    list_near.append(obj)
  return list_near

from signature_detect.loader import Loader
from signature_detect.extractor import Extractor
from signature_detect.cropper import Cropper
from signature_detect.judger import Judger
from flask import Flask,  request, render_template
from werkzeug.utils import secure_filename
import os
UPLOAD_FOLDER = 'static/upload/'
app = Flask(__name__)

@app.route('/detect-signature',methods=['GET', 'POST'])
def detect_signature():
  if request.method == 'POST':
    f = request.files['file']
    path = UPLOAD_FOLDER+'crop.png'
    f.save(path)
    loader = Loader()
    mask = loader.get_masks(path)[0]
    extractor = Extractor(amplfier=15)
    labeled_mask = extractor.extract(mask)
    cropper = Cropper()
    results = cropper.run(labeled_mask)

    if len(results)>0:
      signature = results[0]["cropped_mask"]
      cv2.imwrite(path,signature)
      return render_template('detect.html', check=True)
  return render_template('detect.html', check=False)




@app.route('/',methods=['GET', 'POST'])
def upload_file():
  if request.method == 'POST':
      f = request.files['file']
      path = UPLOAD_FOLDER+'sign.png'
      f.save(path)
      f_dists, f_ids = predict(path)
      list_near = get_object(f_dists, f_ids)
      return render_template('index.html', list_near=list_near)
  else:
    return render_template('index.html')
if __name__ == '__main__':
   app.run(debug = True)