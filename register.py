import cv2
import os
import pickle
from facenet_pytorch import MTCNN, InceptionResnetV1
from numpy.lib.function_base import _flip_dispatcher
import torchvision.models as models
from torchvision import transforms
import torch
from torch.nn.functional import interpolate
import numpy as np
from PIL import Image
import argparse

from torchsummary import summary

from FRNet import FRNet
import config as cfg


ROOT = "./register"

# Device Check
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))


def parse_arg():
    parser = argparse.ArgumentParser(description='Door latch system')
    parser.add_argument('-m', '--model', type=str, help='-m mine/tune/pretrain', default="mine", required=True)
    parser.add_argument('--video', dest='video_name', nargs='?', default="")
    parser.add_argument('--image', dest='img_name', nargs='?', default="")
    parser.add_argument('--register', dest='reg', nargs='?', const=True , default=False)

    args = parser.parse_args()
    return args, parser

def video_writer(vc):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(vc.get(cv2.CAP_PROP_FPS))
    size = (int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter()
    output_file_name = "result.avi"
    writer.open(output_file_name, fourcc, fps, size, True)

    return writer

def inference(filename, model, db_name=cfg.DB_PATH):
    
    # process on single frame
    tfms = transforms.Compose([transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)), transforms.ToTensor()])

    # Create MTCNN
    mtcnn = MTCNN(
        keep_all=True,
        device=device
    )

    # Load Registered encoding Dict
    encoding_dict = load_pickle(file_path=db_name)[0]

    print("filenam: ", filename)

    # Video capture and writer
    vc = cv2.VideoCapture(filename)

    
    if not vc.isOpened():
        print(f"'{filename}' not found")
        return
    else:
        print(f"playing {filename}")

    writer = video_writer(vc)
    
    
    name_count_dict = dict()
    for key in encoding_dict.keys():
        name_count_dict[key] = 0
    print(name_count_dict)

    verify_list = set()
    while True:
        success, frame = vc.read()

        if not success:
            break
        
        boxes, probs = mtcnn.detect(frame, landmarks=False)
        
        # return [x0, y0, x1, y1]
        if probs.any():
            for idx, box in enumerate(boxes):
                if probs[idx] > 0.7:
                    box = list(map(int, box))
                    # print(box)

                    for idx, pt in enumerate(box):
                        # left point
                        if pt < 0:
                            box[idx] = 0
                        
                        if idx % 2 == 0:    # X
                            # greater than width
                            if pt > frame.shape[1]:
                                box[idx] = frame.shape[1] - 1
                        else:   # y
                            if pt > frame.shape[0]:
                                box[idx] = frame.shape[0] - 1

                    face = frame[box[1]:box[3], box[0]: box[2]]
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = tfms(Image.fromarray(face)).to(device).unsqueeze(0)
                    cur_embed = model(face)

                    cv2.rectangle(frame, box[:2], box[2:], (255,0,0), 3)

                    for name, embed in encoding_dict.items():
                        dist = (cur_embed - embed).norm().item()
                        print("Distance from {}: {:3f}".format(name, dist))
                        if dist < 0.9:
                            print(f"Found verified face: {name}")
                            name_count_dict[name] += 1
                            if name_count_dict[name] > 10:
                                verify_list = list(verify_list)
                                verify_list.append(name)
                                verify_list = set(verify_list)
                            break
                        else:
                            pass
            
            height = 30
            print(verify_list)
            if len(verify_list):
                for name in verify_list:
                    cv2.putText(frame, f"VF {name}", (10, height), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0,255,0), 1)
                    height += 30
            else:
                cv2.putText(frame, f"Not Verified", (10, height), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0,0,255), 1)

        cv2.imshow(str(filename), frame)
        writer.write(frame)
        
        if cv2.waitKey(1) == 27:
            break
    
    cv2.destroyAllWindows()
    vc.release()
    writer.release()

def test_img(filename, model, db_name=cfg.DB_PATH):
    # load register DB
    encoding_dict = load_pickle(file_path=db_name)[0]

    # Create MTCNN
    mtcnn = MTCNN(
        keep_all=True,
        device=device
    )

    tfms = transforms.Compose([transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)), transforms.ToTensor()])
    im = cv2.imread(filename)
    
    print(filename)
    print(im.shape)
    # face = mtcnn(im)
    # cur_embed = model(face.to(device))

    img_detect = im.copy()

    boxes, probs = mtcnn.detect(im)
    print(boxes, probs)

    if probs.any():
        # only detect one face
        if probs[0] > 0.7:

            box = list(map(int, boxes[0]))

            for idx, pt in enumerate(box):
                # left point
                if pt < 0:
                    box[idx] = 0
                                    
                if idx % 2 == 0:    # X
                    # greater than width
                    if pt > img_detect.shape[1]:
                         box[idx] = img_detect.shape[1] - 1
                else:   # y
                    if pt > img_detect.shape[0]:
                        box[idx] = img_detect.shape[0] - 1

            face = img_detect[box[1]:box[3], box[0]: box[2]]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = tfms(Image.fromarray(face)).to(device).unsqueeze(0)
            cur_embed = model(face)

            nearest = 20
            people_name = "unknown"
            for name, embeds in encoding_dict.items():
                for embed in embeds:
                    dist = (cur_embed - embed).norm().item()
                    print(f"Distance {dist} away from {name}")
                    if dist < nearest:
                        nearest = dist
                        people_name = name
            
            # draw on detect img
            cv2.rectangle(img_detect, box[:2], box[2:], (255,0,0), 3)
            cv2.putText(img_detect, "verified face: " + people_name, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255,0,0), 1)
    
    print("-" * 20)
    print(people_name)
    cv2.imwrite("result.jpg", img_detect)
    
def load_pickle(file_path=cfg.DB_PATH):
    encoding_dicts = []

    try:
        with open(file_path, 'rb') as f:
            while True:
                try:
                    emb = pickle.load(f)
                    encoding_dicts.append(emb)
                except:
                    break
    except:
        encoding_dicts.append({})
    
    return encoding_dicts    

def save_pickle(obj, file_path=cfg.DB_PATH):
    with open(file_path, "wb") as file:
        pickle.dump(obj, file)

def register(model, output= cfg.DB_PATH):
    
    encoding_dict = dict()
    print("-"*20)
    print(ROOT)
    for root, subdir, files in os.walk(ROOT):
        embeds = []
        error_flag = True
        for f in files:
            try:
                file_name = os.path.join(root, f)
                # img = cv2.imread(file_name)
                img = Image.open(file_name)
                # process on single frame
                tfms = transforms.Compose([transforms.ToTensor()])
                img = tfms(img).unsqueeze(0)

                img = img.to(device)
                embed = model(img)

                embeds.append(embed)
                error_flag = False
            except Exception as error:
                print(error)
                error_flag = True

        if not error_flag:
            print(root)
            label = root.split('/')[2]
            print(label)
            print(len(files))
            # with torch.no_grad():
            #     embeds = sum(embeds) / len(files)
            
            encoding_dict[label] = embeds

    save_pickle(encoding_dict, file_path=output)


if __name__ == "__main__":
    
    args, parser = parse_arg()
    db_names = ["./register/mine-register.pkl","./register/tune-register.pkl", "./register/pre-register.pkl"]
    db_name = ""
    if os.path.exists(cfg.MODEL_PATH):

        # config witch model used
        if args.model == 'mine':
            # Load model
            model = FRNet(pretrained=True, data="WebFace")
            model.load_state_dict(torch.load(cfg.MODEL_PATH))
            
            model.to(device)
            model.eval()
            if args.reg:
                register(model, output=db_names[0])

            db_name = db_names[0]

        elif args.model == 'tune':
            model = InceptionResnetV1(
                classify=True,
                pretrained='vggface2',
                num_classes=428
            )
            model.classify=False

            model.load_state_dict(torch.load("./models/lfw_tune.pth"))
            model.to(device)
            model.eval()
            if args.reg:
                register(model, output=db_names[1])
            db_name = db_names[1]
        else:
            # others pretrained
            model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
            if args.reg:
                register(model, output=db_names[2])
            db_name = db_names[2]

        # summary(model)
        
        if args.video_name:
            filename = os.path.join("./Data/Videos/", args.video_name)
            inference(filename=filename, model=model, db_name=db_name)
        elif args.img_name:
            filename = os.path.join("./Data/Images/", args.img_name)
            test_img(filename=filename, model=model, db_name=db_name)
        else:
            print("only register")

    else:
        print(f"Pretrained Model {cfg.MODEL_PATH} not found")

