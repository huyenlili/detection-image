import torch
import cv2
import os
from PIL import Image
import imagehash
from LAVIS.lavis.models import load_model_and_preprocess
from caption import Summarizer
import os
from mmaction.apis import inference_recognizer, init_recognizer
dic=[{"Bóng đá": ["football", "soccer"]},
    {"Võ thuật": ["kick","punch"]},
    {"Bóng rổ": ["basketball"]},
    {"Cầu lông": ["badminton"]},
    {"Quần vợt": ["tenis"]},
    {"Bóng chuyền": ["volleyball"]},
    {"Bóng bàn": ["table tenis"]},
    {"Gym": ["gym"]},
    {"Đạp xe": []},
    {"Bơi lội": ["swimming"]},
    {"Chạy bộ": [""]},
    {"Game": [""]},
    {"Động vật hoang dã": ["animal"," hippopotamus"," hippo","lion","tiger"]},
    {"Nuôi thú cưng, chăm sóc thú cưng": ["pet", "dog", "cat"]},
    {"Nhảy": ["dancing"]},
    {"Âm nhạc": ["singing", "micro"]},
    {"Nhiếp ảnh": ["photograph"]},
    {"Điêu khắc": ["carving","sculpture"]},
    {"Vẽ tranh": ["paint"]},
    {"Cắm hoa": ["cutting"]},
    {"Làm vườn": ["farm"]},
    {"Thời trang": ["Clothes"]},
    {"Toán, lý, hoá": []},
    {"Thiên văn học": []},
    {"Đồ ăn": ["cooked", "food", "plate", "soup"]},
    {"Đồ uống": []},
    {"Người ngoài hành tinh": []},
    {"Ô tô": ["car, driver, driving"]},
    {"Xe máy":["motorcycles", "motor"]}
]
device="cpu"
md, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)
def delete_file(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
        except Exception as e:
            print(f'Error deleting {file_path}: {e}')
def detect_video(video_path, tempdir):
    cam = cv2.VideoCapture(video_path)
    try:
        if not os.path.exists('data_images'):
            os.makedirs('data_images')
    except OSError:
        print ('Error: Creating directory of data')
    currentframe = 0
    import time
    valid = True
    captions = []
    step_second = 0
    while(True):
        t_msec = 1000*(step_second)
        cam.set(cv2.CAP_PROP_POS_MSEC, t_msec)
        ret, frame = cam.read()
        ret,frame = cam.read()
        if ret:
            name = f'{tempdir}/{str(currentframe)}.jpg'
            cv2.imwrite(name, frame)
            raw_image = Image.open(name).convert("RGB")
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
            captions += md.generate({"image": image}, use_nucleus_sampling=True, num_captions=4)
            currentframe += 1
        else:
            break
        step_second +=2
    return valid, captions
if __name__ == '__main__':
    valid, captions = detect_video('2 (1).mp4', 'temp')
    print(valid, captions)
    # config = 'mmaction2/configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py'
    # checkpoint = 'mmaction2/checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth'
    # model = init_recognizer(config, checkpoint, device='cpu')
    # label = 'mmaction2/tools/data/kinetics/label_map_k700.txt'
    # kq=[]
    # dir="data_images"
    # z=0
    # for i in range(1,2): 
        # print("Clip so", i)
        # video = 'dataset/'+str(i)+".mp4"
        # results = inference_recognizer(model, video)
        # labels = open(label).readlines()
        # labels = [x.strip() for x in labels]
        # results = [(labels[k[0]], k[1]) for k in results]
        # if "shooting goal" in results[0][0] or (results[0][0]=="hurling (sport)" and results[1][0]=="shooting goal (soccer)"):
        #     kq.append("Bong da")
        # elif results[0][0]=="dunking basketball":
        #     kq.append("Bong ro")
        # elif results[0][0]=="playing volleyball":
        #     kq.append("Bong chuyen")
        # elif results[0][0]=="playing tennis":
        #     kq.append("tennis")
        # elif results[0][0] == "playing badminton":
        #     kq.append("cau long")
        # elif results[0][0] == "swimming":
        #     kq.append("Swimming")
        # elif results[1][0]=="playing basketball":
        #     kq.append("Bong ro")
        # elif "dancing" in results[0][0]:
        #     kq.append("Nhay")
        # elif ("singing" in results[0][0]):
        #     kq.append("Hat")
        # elif ("dog" in results[0][0] or "cat" in results[0][0]) or ("dog" in results[1][0] or "cat" in results[1][0]):
        #     kq.append("Thu cung")
        # elif results[0][0]=="driving car":
        #     kq.append("Car")
        # elif results[0][0]=="riding a bike":
        #     kq.append("Xe dap")
        # elif "kick" in results[0][0] or "punching" in results[0][0]:
        #     kq.append("Vo thuat")
        # elif results[0][0]=="motorcycling":
        #     kq.append("Xe may")
        # elif "jump" in results[0][0]:
        #     kq.append("Dien kinh")
        # else:
        #     delete_file(dir)
        #     valid, captions = detect_video(video,dir)
        #     summ =  Summarizer(language='en', summary_length=4)
        #     txt=summ.summarize(captions).split(" ")
        #     br=False
        #     for g in dic:
        #         for k in txt:
        #             for h in g.keys():
        #                 if k in g[h]:
        #                     kq.append(h)
        #                     br=True
        #                     break
        #             if br:
        #                 break 
        #         if br:
        #             break 
        #     if br==False:
        #         kq.append("pass")
        # print(kq[z])
        # z=z+1
