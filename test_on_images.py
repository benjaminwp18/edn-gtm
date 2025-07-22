import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import cv2
import numpy as np

from core.utils import deprocess_image, get_file_name
from core.networks import unet_spp_large_swish_generator_model
from core.dcp import estimate_transmission
from core.config import IMG_SIZE

OUTPUT_DIR = "outputs/Dense2"
# IMG_SRC = glob.glob("/home/mantis/Documents/ms/data/dense-haze/*.jpg")
# IMG_SRC = ('/home/mantis/Documents/ms/data/dense-haze/hazy/01_hazy.png',)
IMG_SRC = ("/home/mantis/Videos/cleaned/frames/png/LT_65_100_frame.png",)
WEIGHT_PATH = "weights/densehaze_generator_in512_ep85_loss227.h5"


def preprocess_cv2_image(cv_img):
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    cv_img = cv2.resize(cv_img, (IMG_SIZE, IMG_SIZE))
    img = np.array(cv_img)
    img = (img - 127.5) / 127.5
    return img


def preprocess_depth_img(cv_img):
    cv_img = cv2.resize(cv_img, (IMG_SIZE, IMG_SIZE))
    img = np.array(cv_img)
    img = np.reshape(img, (IMG_SIZE, IMG_SIZE, 1))
    img = 2 * (img - 0.5)
    return img


g = unet_spp_large_swish_generator_model()
g.load_weights(WEIGHT_PATH)
g.summary()

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

if __name__ == "__main__":
    for cnt, img_path in enumerate(IMG_SRC):
        img_name = get_file_name(img_path)
        ori_image = cv2.imread(img_path)

        if ori_image is None:
            print(f"Failed to load image: {ori_image}")
            continue

        h, w, _ = ori_image.shape

        # ori_image_resized = cv2.resize(ori_image, (img_size,img_size))
        # cv2.imwrite(f"{img_name}_resized.jpg", ori_image_resized)

        t = estimate_transmission(ori_image)
        t = preprocess_depth_img(t)

        ori_image = preprocess_cv2_image(ori_image)

        x_test = np.concatenate((ori_image, t), axis=2)

        x_test = np.reshape(x_test, (1, IMG_SIZE, IMG_SIZE, 4))
        generated_images = g.predict(x=x_test)

        de_test = deprocess_image(generated_images)
        de_test = np.reshape(de_test, (IMG_SIZE, IMG_SIZE, 3))

        # pred_image_resized = cv2.cvtColor(de_test, cv2.COLOR_BGR2RGB)
        # cv2.imwrite(f"{img_name}_resized_pred.jpg", pred_image_resized)

        de_test = cv2.resize(de_test, (w, h))

        rgb_de_test = cv2.cvtColor(de_test, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"{OUTPUT_DIR}/{img_name}.jpg", rgb_de_test)

        print(cnt, len(IMG_SRC))
        # if cnt==10: break

    print("Done!")
