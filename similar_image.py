import cv2
import os
from PIL import Image
import imagehash
from lavis.models import load_model_and_preprocess
import torch
# import argostranslate.package
# import argostranslate.translate


from_code = "en"
to_code = "vi"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model, vis_processors, _ = load_model_and_preprocess(
    name="blip_caption", model_type="base_coco", is_eval=True, device=device
)

# def translate(text, from_code = 'en', to_code = 'vi'):
#     return argostranslate.translate.translate(text, from_code, to_code)


def detect_video(video_path, tempdir):
    # Read the video from specified path
    print(1)
    cam = cv2.VideoCapture(video_path)
  
    try:
        
        # creating a folder named data
        if not os.path.exists('data_images'):
            os.makedirs('data_images')
    
    # if not created then raise error
    except OSError:
        print ('Error: Creating directory of data')
    
    # frame
    currentframe = 0

    import time
    start_time = time.time()

    valid = True
    captions = []
    while(True):
    # reading from frame
        ret,frame = cam.read()
    
        if ret:
            # if video is still left continue creating images
            name = f'{tempdir}/{str(currentframe)}.jpg'
        
            # writing the extracted images
            cv2.imwrite(name, frame)
        
            # similar
            if currentframe> 0:
            
                hash0 = imagehash.average_hash(Image.open(name)) 
                similar = False
                for i in range(0, currentframe):
                    last_name = f'{tempdir}/{str(i)}.jpg'
                    hash1 = imagehash.average_hash(Image.open(last_name)) 
                    cutoff = 5  # maximum bits that could be different between the hashes. 
                    if hash0 - hash1 < cutoff:
                        similar = True
                        break
                if similar:
                    continue
        
            raw_image = Image.open(name).convert("RGB")
          
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
            
            start_time = time.time()
            captions += model.generate({"image": image}, use_nucleus_sampling=True, num_captions=20)
            valid, suspect = valid_text(captions)
            if suspect:
                captions = model.generate({"image": image}, use_nucleus_sampling=True, num_captions=20)
                valid, suspect = valid_text(captions)
            if not valid:
                break
            print("--- %s seconds ---" % (time.time() - start_time))
            # results = map(translate, captions)
            currentframe += 1
        else:
            break
        
    return valid, captions

def detect_image(image_path):
    raw_image = Image.open(image_path).convert("RGB")

    # we associate a model with its preprocessors to make it easier for inference.
        
    # uncomment to use base model
   
    vis_processors.keys()

    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    # print(model.generate({"image": image}))
    contents = model.generate({"image": image}, use_nucleus_sampling=True, num_captions=20)
    # print(contents)
    valid, suspect = valid_text(contents)
    print(contents)
    if not valid:
        return valid, contents
    
    if suspect:
        contents = model.generate({"image": image}, use_nucleus_sampling=True, num_captions=30)
        valid, suspect = valid_text(contents)
    return valid, contents

def valid_text(contents):
    text = ' '.join(contents)
    suspect_word = ['breast', ' chest', 'sexy', 'nude',' ass ', 'panties', 'bikini', 'naked', ' thong ']
    bad_word = ['sucking','fuck','dick', 'cock', 'anal','porn', 'xvideo', 'pussy', 'sex ', 'orgasm', 'handjob','blowjob']
    suspect = False
    find_badword = [s for s in bad_word if s in text]
    find_suspect = [s for s in suspect_word if s in text]
    
    print("find_badword", find_badword)
    print("suspect_word", find_suspect)
    if len(find_suspect) > 0:
       suspect = True 
    if len(find_badword) > 0:
        return False, suspect
    elif  len(find_suspect) > 0  and ('bikini' in find_suspect):
        return True, suspect
    else:
        return len(find_badword + find_suspect) == 0, suspect
    



if __name__ == '__main__':
    video_path = '/home/emso/LAVIS/GAPO.mp4'
    dir_path = 'data_images'
    valid, captions = detect_video(video_path, dir_path)
    print(valid, captions)