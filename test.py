import torch
import cv2
import os
from PIL import Image
import imagehash
from lavis.models import load_model_and_preprocess
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
    {"Leo núi":["hill","climb"]},
    {"Đạp xe": ["bicycle"]},
    {"Bơi lội": ["swimming"]},
    {"Chạy bộ": []},
    {"Game": ["game"]},
    {"Động vật hoang dã": ["animal"," hippopotamus"," hippo","lion","tiger","pig"]},
    {"Nuôi thú cưng, chăm sóc thú cưng": ["pet", "dog", "cat"]},
    {"Nhảy": ["dancing"]},
    {"Âm nhạc": ["singing"]},
    {"Nhiếp ảnh": ["photo"]},
    {"Điêu khắc": ["carving","sculpture"]},
    {"Vẽ tranh": ["paint","draw"]},
    {"Cắm hoa": ["cutting"]},
    {"Làm vườn": ["farm"]},
    {"Toán, lý, hoá": ["graph","math"]},
    {"Thiên văn học": ["space","earth","moon","star"]},
    {"Đồ ăn": ["cooked", "food", "plate", "soup","eat"]},
    {"Đồ uống": ["drink"]},
    {"Người ngoài hành tinh": ["alien"]},
    {"Ô tô": ["car","drive", "driving"]},
    {"Xe máy":["motorcycle", "motor"]},
    {"Thời trang": ["cloth","purse"]},
    {"Chính trị":["rocket","gun"]}
]
mact=[
    {"skip":["beer pong","treating wood","lifting hat","pushing cart","installing carpet","carrying weight","disc golfing","not baby","hunting","pushing wheelchair","falling off chair","faceplanting"]},
    {"Thể thao trên biển":["jetskiing","surfing","sailing","docking boat","water skiing","water sliding","scuba","cano"]},
    {"Hoạt động trên tuyết":["skiing","snow","bobsledd"]},
    {"Bóng chày":["baseball","cricket","catching"]},
    {"Cử tạ":["lifting"]},
    {"Đồ ăn":["salad","meat","cheese","rolling pastry","marshmallow","onion","oysters","dining","fish","vegetable","banana","blueberrie","potato","bread","refrige","eggs","sausage","eating","cooking","pig","cookie","cake","food","barbequing","sandwich","pizza","sushi","fruit","hotdog"]},
    {"Nuôi thú cưng, chăm sóc thú cưng":["herding","sheep","camel","cat","zoo","dog","petting","feed","horse","cow","goat","mule","elephant"]},
    {"Bóng đá":["soccer","kickball","hurling","kicking field goal"]},
    {"Gym":["jogging","hula hooping","head stand","cartwheeling","lunge","running on treadmill","squat","hula hooping","stretching","exercis","yoga","aerobic","gym","tai chi","training","bench pressing","bending back","standing on hands"]},
    {"Bóng rổ":["basketball"]},
    {"Bóng chuyền":["volleyball"]},
    {"Khúc côn cầu":["hockey"]},
    {"Tenis":["tennis","racquetball","ping pong"]},
    {"Cầu lông":["badminton"]},
    {"Bơi":["snorkeling","bodysurf","swim","diving cliff"]},
    {"Nhảy":["pirouetting","cumbia","moon walking","ballet","disco","danc","zumba"]},
    {"Cờ vua":["monopoly","chess","checker","maracas"]},
    {"Thẻ bài":["card","poker","blackjack","mahjong"]},
    {"Vẽ":["colour","drawing","graffiti","paint","spray"]},
    {"Hát":["karaoke","busking","singing"]},
    {"Bowling":["bowling"]},
    {"Điêu khắc":["pottery","chiseling","carving","wood burning (art)"]},
    {"Nghệ thuật":["acting","calligraphy","tattoo"]},
    {"Đồ uống":["pulling espresso shot","coffee","bottle","latte art","grape","cup","drinking","coconut","watermelon","pineapple","orange","apple","beer","wine","bartending","tea","champagne"]},
    {"Bóng bầu dục":["football"]},
    {"Leo núi":["abseiling","climb"]},
    {"Bi-a":["billiard"]},
    {"Thiên văn học":["zero gravity"]},
    {"Cắm hoa":["flower"]},
    {"Âm nhạc":["recorder","lute","harmonica","didgeridoo","cymbal","clarinet","hula hooping","music","beatbox","tapping","drum","guitar","violin","accordion","trumpet","saxophone","piano","organ","bagpipes","cello","xylophone","ukulele","trombone","ocarina"]},
    {"Xe máy":["moto","segway"]},
    {"Làm đẹp":["hair","lipstick","mascara","shaving","decoupage","beard","scrubbing","waxing","nail","applying cream","eye"]},
    {"Bắn cung":["archery","axe","knife"]},
    {"Lịch sử":["historical reenactment","building sandcastle","archaeological excavation"]},
    {"Đấu vật":["wrest"]},
    {"Máy tính":["computer"]},
    {"Xe đạp":["bicycle","bike"]},
    {"Điền kinh":["pole vault","hurdling","para","parkour","skydiving","hoverboarding","skat","trapezing","jump","javelin","hammer throw","backflip"]},
    {"Nhiếp ảnh":["taking photo"]},
    {"Võ thuật":["fencing","sword","capoeira","punching","kick"]},
    {"Game":["domino","puzzle","rubik","slot machine","controller"]},
    {"Golf":["golf"]},
    {"Làm vườn":["bee keeping","plant","tree","shrub"]},
    {"Xiếc":["tightrope walking","stilt"]},
    {"Cơ khí":["making the bed","jewelry","repair","knive","felt","blowing glass","metal","welding","wrench","saw","hammer","power drill"]},
    {"Bi đá":["curling (sport)"]},
    {"Quân sự":["saluting","marching"]},
    {"Sinh học":["microscope"]},
    {"Review":["unbox"]},
    {"Sức khỏe":["inhaler","banda","massa","cracking"]},
    {"Trẻ em":["baby","puppet","child"]},
    {"Chính trị":["attending conference"]},
    {"Pháp luật":["testifying"]},
    {"Thời trang":["sari","shoes","knitting","crocheting","embroider","laundry","tie dying","sewing","tying","weav","ironing","cloth","yarn spinning","needle"]},
    {"Ô tô":["car","tire"]},
    {"Thời sự":["news","weather"]}
]
device="cuda"
md, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)
# kiểm tra các label sau khi tính toán ra thuộc nhóm gì
def compare(label,dic):
    br=False
    for value_label in label:
        for list_dic in dic:
            for key_dic in list_dic.keys():
                for value_list_dic in list_dic[key_dic]:
                    if value_list_dic in value_label:
                        if key_dic!="skip":
                            return key_dic
                        else:
                            br=True
                            break
                if br:
                    break
            if br:
                br=False
                break
    return "skip"
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
    config = 'mmaction2/configs/recognition/tsn/tsn_r50_video_1x1x8_100e_kinetics700_rgb.py'
    checkpoint = 'mmaction2/checkpoints/tsn_r50_video_1x1x8_100e_kinetics700_rgb_20201015-e381a6c7.pth'
    model = init_recognizer(config, checkpoint, device='cuda')
    label = 'mmaction2/tools/data/kinetics/label_map_k700.txt'
    kq=[]
    dir="data_images"
    z=0
    for i in range(227,228): 
        print("Clip so", i)
        video = 'dataset/'+str(i)+".mp4"
        results = inference_recognizer(model, video)
        labels = open(label).readlines()
        labels = [x.strip() for x in labels]
        results = [(labels[k[0]], k[1]) for k in results]
        lb=[]
        for k in results:
            if(k[1]>=7.2):
                lb.append(k[0])
            else:
                break
        rs=compare(lb,mact)
        if rs!="skip":
            kq.append(rs)
        else:
            delete_file(dir)
            valid, captions = detect_video(video,dir)
            summ =  Summarizer(language='en', summary_length=4)
            txt=summ.summarize(captions).split(" ")
            print(summ.summarize(captions))
            rs=compare(txt,dic)
            kq.append(rs)
        for result in results:
            print(f'{result[0]}={result[1]}')
        print("++++++++",kq[z],"++++++++")
        z=z+1
