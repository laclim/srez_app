# app.py
#
# A simple example of hosting a TensorFlow model as a Flask service
#
# Copyright 2017 ActiveState Software Inc.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import logging
import random
import time
import srez_main
from flask import Flask, jsonify, request,render_template
from flask_sqlalchemy import SQLAlchemy
import base64
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from PIL import Image
import io
UPLOAD_FOLDER = '/static/'
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///img.db'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

db = SQLAlchemy(app)




class Img(db.Model):
  id = db.Column(db.Integer,primary_key=True)
  name = db.Column(db.String(300))
  data=db.Column(db.LargeBinary)

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

# def load_labels(label_file):
#   label = []
#   proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
#   for l in proto_as_ascii_lines:
#     label.append(l.rstrip())
#   return label



@app.route('/')
def index():
    
    with tf.Session() as sess:
      [gene_minput, gene_moutput, gene_output, gene_var_list, disc_real_output, disc_fake_output, disc_var_list] = srez_main.init_model(sess)

    # file_name = request.args['file']

    # # t = read_tensor_from_image_file(file_name,
    # #                               input_height=input_height,
    # #                               input_width=input_width,
    # #                               input_mean=input_mean,
    # #                               input_std=input_std)
        
    # with tf.Session(graph=graph) as sess:
    #     start = time.time()
    #     results = sess.run(output_operation.outputs[0],
    #                   {input_operation.outputs[0]: t})
    #     end=time.time()
    #     results = np.squeeze(results)

    #     top_k = results.argsort()[-5:][::-1]
    #     labels = load_labels(label_file)

    # print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))
    # srez_main._demo(filenames=file_name)
    # for i in top_k:
    #     print(labels[i], results[i])
   
    return render_template('index.html')
    # return jsonify(labels,results.tolist())

# @app.route('/')
# def gan():
#   file = request.files['avatar']
#   srez_main._demo(filenames=file_name)
def getFilename():
  obj = Img.query.order_by(Img.id.desc()).all()
  last=obj[0]
  
  FINAL_IMG="finalimg"+str(last.id+1)+".png"
  print(FINAL_IMG)
  # render_template('index.html',final=FINAL_IMG)
  return FINAL_IMG
    


@app.route('/upload', methods=['POST','GET'])
def upload():
  file = request.files['avatar']
  dataimg=file.read()
  # sfile=request.files['image_data']
  filename = secure_filename(file.filename)
  image = Image.open(io.BytesIO(dataimg))
  image.save('static/abc.png')
  newFile=Img(name=file.filename,data=dataimg)
  
  # file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.read()))
  
  data={'filename':filename}
  db.session.add(newFile)
  db.session.commit()
  srez_main._demo('static/abc.png')
  # db.session.refresh(obj)
  FINAL_IMG=getFilename()
  data = {'image': FINAL_IMG}
  
  return jsonify(data)
  
  

@app.route('/get', methods=['GET'])
def get():
  img = Img.query.get(2)
  return render_template('index.html' , final='sadas')

if __name__ == '__main__':
    # TensorFlow configuration/initialization
    # model_file = "checkpoint/checkpoint_new.txt.meta"
    # label_file = "retrained_labels.txt"
    # input_height = 224
    # input_width = 224
    # input_mean = 128
    # input_std = 128
    # input_layer = "input"
    # output_layer = "final_result"

    # # Load TensorFlow Graph from disk
    # graph = load_graph(model_file)

    # # Grab the Input/Output operations
    # input_name = "import/" + input_layer
    # output_name = "import/" + output_layer
    # input_operation = graph.get_operation_by_name(input_name);
    # output_operation = graph.get_operation_by_name(output_name);

    # Initialize the Flask Service
    # Obviously, disable Debug in actual Production
    app.run(debug=True, port=8000)

