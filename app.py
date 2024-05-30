#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install paddlepaddle


# In[2]:


pip install --user "paddleocr>=2.0.1"


# In[3]:


pip install flask boto3 pdf2image paddleocr Pillow


# In[ ]:


import webbrowser
from flask import Flask, request, jsonify
from PIL import Image
import os
import boto3
from pdf2image import convert_from_path
from paddleocr import PaddleOCR

app = Flask(__name__)

ocr_ar = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, use_mkl=True, enable_mkldnn=False,  ocr_version='PP-OCRv4')


def getOCRtext(result_ar):
    state, propa = [], []
    for idx in range(len(result_ar)):
        res = result_ar[idx]
        for line in res:
            state.append(line[1][0])#if are_archar(line[1][0]) else line[1][0])
            propa.append(line[1][1])
    dectext = ' '.join(state)
    propability = sum(propa)/len(propa)
    return  dectext, propability
def download_from_s3(bucket, key, local):
    s3 = boto3.client('s3')
    s3.download_file(bucket, key, local)

def convert_to_jpg(image_path):
    image = Image.open(image_path)
    converted_image_path = image_path.replace('.png', '.jpg').replace('.jpeg', '.jpg')
    image.convert('RGB').save(converted_image_path, 'JPEG')
    return converted_image_path

def process_S3_doc(bucket, key):
    temp_dir = '/tmp'

    local_path = os.path.join(temp_dir, key.replace('/', '_'))

    try:
        download_from_s3(bucket, key, local_path)
    except Exception as e:
        return f'Error downloading file: {e}', 500

    imgpath = None

    # Check file extension and convert if needed
    if key.lower().endswith(('.png', '.jpeg', '.jpg')):
        imgpath = convert_to_jpg(local_path)
    elif key.lower().endswith('.pdf'):
        try:
            images = convert_from_path(local_path, first_page=1, last_page=1)
            imgpath = os.path.join(temp_dir, 'converted.jpg')
            images[0].convert('RGB').save(imgpath, 'JPEG')
        except Exception as e:
            return f'Error converting PDF: {e}', 500

    if imgpath:
        try:
            result = ocr_ar.ocr(imgpath, cls=True)
            text, prop = getOCRtext(result)
            return text, prop
        except Exception as e:
            return f'Error running OCR: {e}', 500
    else:
        return 'Unsupported file type', 400




    result = ocr_ar.ocr(imgpath, cls=True)
    text, prop = getOCRtext(result)
    return text, prop

@app.route('/')
def hello_world():
    return 'Hello'

@app.route('/paddle-ocr', methods=["POST"])
def process_image():
    input_json = request.get_json(force=True)
    bucket = input_json['bucket']
    key = input_json['key']

    text, prop = process_S3_doc(bucket, key)
    print(f'Text: {text}, Accuracy: {prop}')

    return jsonify({
        'statusCode': 200,
        'body': {
            'text': text,
            'accuracy': prop
        }
    })

if __name__ == '__main__':
    # Start the Flask application
    app.run(host='0.0.0.0', port=80, threaded=False)

    # Open the default web browser to the Flask application URL
    webbrowser.open('http://localhost')


# In[ ]:




