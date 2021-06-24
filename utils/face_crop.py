# from mtcnn import MTCNN
from facenet_pytorch import MTCNN
import os
import cv2
import torch
import numpy as np

# NEW_ROOT = "./FACE"
# ROOT = "./LFW"
# ROOT = "CASIA-WebFace"
# NEW_ROOT = "./WebFace"

ROOT = "../register/obama"
NEW_ROOT = "../register/"


# def crop_face(img, face):
#     x, y, w, h = face
#     crop_img = img[y:y+h, x:x+w]
#     crop_img = cv2.resize(crop_img, (250, 250))
#     return crop_img


if __name__ == "__main__":
    print(os.listdir(ROOT))

    # Device Check
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    mtcnn = MTCNN(
        post_process=False,
        device=device
    )

    if not os.path.exists(NEW_ROOT):
        os.mkdir(NEW_ROOT)

    for root, subdir, files in os.walk(ROOT):
        print("-------------")
        print(root)
        for f in files:
            if f.find("."):
                file_name = os.path.join(root, f)
                img = cv2.imread(file_name)

                dir_name = root.split("/")[-1]
                filename = os.path.join(NEW_ROOT, dir_name, f)
                print(filename)

                new_dir = os.path.join(NEW_ROOT, dir_name)
                if not os.path.exists(new_dir):
                    print("create " + new_dir)
                    os.mkdir(new_dir)

                try:
                    # detector = MTCNN()
                    # detect_face = detector.detect_faces(img)
                    detect_face = mtcnn(img)

                    # # return [x, y, width, height]
                    # face = detect_face[0]['box'] 
                    # print(face)

                    # face = crop_face(img, face)
                    # cv2.imwrite(filename, face)

                    face = detect_face.permute(1, 2, 0).int().numpy()

                    cv2.imwrite(filename, face)
                except  Exception as e:
                    print(f"error: {e}")

    cv2.waitKey(0)
