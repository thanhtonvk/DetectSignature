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


from flask import Flask,  request, render_template
from werkzeug.utils import secure_filename
import os
UPLOAD_FOLDER = 'static/upload/'
app = Flask(__name__)
@app.route('/',methods=['GET', 'POST'])
def upload_file():
  if request.method == 'POST':
      f = request.files['file']
      path = UPLOAD_FOLDER+'sign.png'
      f.save(path)
      f_dists, f_ids = predict(path)
      list_near = get_object(f_dists, f_ids)
      print(list_near)
      return render_template('index.html', list_near=list_near)
  else:
    return render_template('index.html')
if __name__ == '__main__':
   app.run(debug = True)