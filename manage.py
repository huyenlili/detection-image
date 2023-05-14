from flask import Flask, request, jsonify
from similar_image import detect_image, detect_video

import requests
import os
import shortuuid
import shutil

labels = [
    {"Thời sự": []},
    {"Thế giới": []},
    {"Pháp luật": []},
    {"Bóng đá": ["football", "soccer"]},
    {"Võ thuật": []},
    {"Bóng rổ": []},
    {"Cầu lông": []},
    {"Quần vợt": ["tenis"]},
    {"Bóng chuyền": []},
    {"Bóng bàn": ["table tenis"]},
    {"Gym": ["gym"]},
    {"Đạp xe": [""]},
    {"Bơi lội": [""]},
    {"Chạy bộ": [""]},
    {"Game": [""]},
    {"Chứng khoán": []},
    {"Bất động sản": []},
    {"Ngân hàng": []},
    {"Tài chính": []},
    {"Ngôn ngữ lập trình": []},
    {"Đồ điện": []},
    {"Tin công nghệ": []},
    {"Viễn thông": []},
    {"Điện thoại": []},
    {"Máy tính": []},
    {"Động vật hoang dã": []},
    {"Nuôi thú cưng, chăm sóc thú cưng": ["pet", "dog", "cat"]},
    {"Diễn viên, người mẫu, ca sĩ": []},
    {"Nhảy": ["dancing"]},
    {"Âm nhạc": ["singing", "micro"]},
    {"Phim": []},
    {"Nhiếp ảnh": ["photograph"]},
    {"Điêu khắc": []},
    {"Vẽ tranh": []},
    {"Cắm hoa": []},
    {"Làm vườn": []},
    {"Thời trang": ["Clothes"]},
    {"Toán, lý, hoá": []},
    {"Ngoại ngữ": []},
    {"Lịch sử": []},
    {"Thiên văn học": []},
    {"Tâm lý học": []},
    {"Nấu ăn": ["cooked", "food", "plate", "soup"]},
    {"Kỹ năng sống": []},
    {"Mẹo vặt cuộc sống": []},
    {"Văn học": []},
    {"Khoa học tự nhiên": []},
    {"Làm đẹp": []},
    {"Chế độ dinh dưỡng": []},
    {"Bài thuốc dân gian": []},
    {"Các bệnh thường gặp": []},
    {"Đồ ăn": ["cooked", "food", "plate", "soup"]},
    {"Đồ uống": []},
    {"Người ngoài hành tinh": []},
    {"Bói toán, lá số tử vi, kinh dịch": []},
    {"Tôn giáo": []},
    {"Địa điểm du lịch": ["scenic scenery", "ocean", "the beach", "grassy hill"]},
    {"Tour": []},
    {"Review nhà hàng, khách sạn": []},
    {"Ô tô": ["car, driver, driving"]},
    {"Xe máy":["motorcycles", "motor"]}
]

app = Flask(__name__)

@app.route('/predict', methods=['POST'])

def predict_media():
    data = request.json
    valid = True
    classification = None
    detection = None
    captions = []
    tempdir = f'tmp/{shortuuid.uuid()}'
    os.makedirs(tempdir) 
    try:
        if 'image_urls' in data:
            for image_url in data['image_urls']:
                image_path = f'{tempdir}/{shortuuid.uuid()}.jpg'
                img_data = requests.get(image_url).content
                with open(image_path, 'wb') as handler:
                    handler.write(img_data)
                
                valid, captions = detect_image(image_path)
                if not valid:
                    break
                
        if 'video_urls' in data and valid:    
            for video_url in data['video_urls']:
                image_path = f'{tempdir}/{shortuuid.uuid()}.mp4'
                r=requests.get(video_url)
                f=open(image_path,'wb')
                for chunk in r.iter_content(chunk_size=255): 
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
                f.close()
                
                valid, captions = detect_video(image_path, tempdir)
                if not valid:
                    break
            
    except Exception as e:
        raise e
        return jsonify(error=422, text=str(e)), 422
    finally:
       shutil.rmtree(tempdir) 
    
    print(valid, captions)
    return jsonify({'valid': valid,
                    'captions': captions,
                    })

# @app.route('/classification_video', methods=['POST'])
# def classification_video():
#     data = request.json
#     classification = None
#     tempdir = f'tmp/{shortuuid.uuid()}'
#     video_path = data['path']
    
#     captions = detect_video(video_path, tempdir)
#     from caption import Summarizer
#     summ =  Summarizer(language='en', summary_length=3)
#     text = summ.summarize(captions)
#     return jsonify(classification)
    

if __name__ == '__main__':
    app.run(debug=False, port=5002, host = '0.0.0.0')
    # print(detect_image('/home/emso/LAVIS/2efeaeca-1b1d-4ca1-8291-ac9bd93cc92c.jpg'))
