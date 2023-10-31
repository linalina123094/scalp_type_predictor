<!DOCTYPE html>
<html>
<head>
    <title>두피빡빡 홈페이지</title>
    <style>
        .interface-container {
            display: flex;
            flex-direction: column; /* 세로로 배치하도록 변경 */
        }
        .interface {
            flex: 1;
            padding: 10px;
            margin: 10px;
            border: 1px solid #ddd;
        }
    </style>
</head>
<body>
    <h1>Integrated Gradio Interfaces</h1>
    <div class="interface-container">
        <div class="interface">
            <!-- Embed your first Gradio interface here -->
            <iframe src="호스팅 사이트에서 제공하는 url 입력" width="100%" height="500"></iframe>
        </div>
        <div class="interface">
            <!-- Embed your second Gradio interface here -->
            <iframe src="호스팅 사이트에서 제공하는 url 입력" width="100%" height="500"></iframe>
        </div>
        <div class="interface">
            <!-- Embed your third Gradio interface here -->
            <iframe src="호스팅 사이트에서 제공하는 url 입력" width="100%" height="500"></iframe>
        </div>
    </div>
</body>
</html>

!pip install gradio
import gradio as gr
import keras
import numpy as np
from PIL import Image, ImageOps

# 사용자 정의 함수
def greet (img ):
   data = np.ndarray(shape=(1 , 224 , 224 , 3 ), dtype=np.float32)
   image = Image.fromarray(img).convert("RGB")
   size = (224 , 224 )
   image = ImageOps.fit(image, size, Image.ANTIALIAS)
   image_array = np.asarray(image)
   normalized_image_array = (image_array.astype(np.float32) / 127.5 ) - 1
   data[0 ] = normalized_image_array
   prediction = model.predict(data)
   index = np.argmax(prediction)
   return class_names[index]

# 모델 및 클래스명 정의
model = keras.models.load_model('keras_model.h5')
class_names = ['탈모성 두피', '건조성 두피(미세각질)', '지성두피(피지과다)', '홍반성 두피(모낭홍반농포)', '정상 두피']

# Gradio 인터페이스 생성
iface = gr.Interface(
   fn=greet,
   inputs=gr.inputs.Image(),
   outputs='text',
   title="두피빡빡",  # 제목 설정
   description="촬영한 두피 사진을 업로드하세요.",
)

# 웹 앱 실행
iface.launch(share=True )

!pip install --upgrade pip
!pip install -q openai gradio

import gradio as gr
import openai

api_key = "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"
openai.api_key = api_key

def predict (input , history ):
   history.append({"role": "user", "content": input})

   try :
     gpt_response = openai.ChatCompletion.create(
       model="gpt-3.5-turbo",
       messages=history
     )
     response = gpt_response["choices"][0 ]["message"]["content"]
     history.append({"role": "assistant", "content": response})

   except Exception as e:
     response = f "An error occurred: {str (e)}"
     history.append({"role": "assistant", "content": response})

   messages = [(history[i]["content"], history[i+1 ]["content"]) for i in range (1 , len (history), 2 )]

   return messages, history

with gr.Blocks() as demo:
   chatbot = gr.Chatbot(label="ChatBot")

   state = gr.State([{
     "role": "system",
     "content": "당신은 친절한 인공지능 챗봇입니다. 질문에 대해 간결하고 친절하게 답변합니다."
   }])

   with gr.Row():
     txt = gr.Textbox(show_label=False , placeholder="챗봇에게 무엇을 물어보시겠습니까?").style(container=False )

   txt.submit(predict, [txt, state], [chatbot, state])

demo.launch(debug=True , share=True )

!pip install gradio
!pip install --upgrade pip
!pip install -q openai gradio
!pip install flask


import os
import sys
import urllib.request
import pandas as pd
import json
import re
import gradio as gr
from urllib.parse import quote

client_id = "Z0kUR0ZW5Sc4YqJnYf6f"
client_secret = "pwgB_h9xFu"

def search_on_naver (query ):
   idx = 0
   display = 100
   start = 1
   end = 100

   # 질의어를 URL 인코딩합니다.
   query_encoded = quote(query)

   shopping_df = pd.DataFrame(columns=["Title", "Link"])

   for start_index in range (start, end, display):
     url = f "https://openapi.naver.com/v1/search/shop.json?query={query_encoded}" \
       + f "&display={display}" \
       + f "&start={start_index}"

     request = urllib.request.Request(url)
     request.add_header("X-Naver-Client-Id", client_id)
     request.add_header("X-Naver-Client-Secret", client_secret)
     response = urllib.request.urlopen(request)
     rescode = response.getcode()

     if rescode == 200 :
       response_body = response.read()
       response_dict = json.loads(response_body.decode('utf-8'))
       items = response_dict['items']
       for item_index in range (0 , len (items)):
         remove_tag = re.compile ('<.*>')
         title = re.sub(remove_tag, '', items[item_index]['title'])
         link = items[item_index]['link']
         shopping_df.loc[idx] = [title, link]
         idx += 1
     else :
       print ("Error Code:" + rescode)

   return shopping_df.to_string()

iface = gr.Interface(
   fn=search_on_naver,
   inputs="text",
   outputs="text",
   title="제품 검색",
   description="두피 관련 제품을 검색하세요."
)

iface.launch(debug=True )
