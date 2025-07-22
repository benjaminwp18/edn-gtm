import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import cv2
import glob
import numpy as np
import math
import timeit

from core.utils import deprocess_image
from core.networks import unet_spp_large_swish_generator_model
from core.dcp import estimate_transmission
from core.utils import get_file_name
from core.config import RESHAPE, IMG_SIZE


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))


def preprocess_cv2_image(cv_img):
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    cv_img = cv2.resize(cv_img, RESHAPE)
    img = np.array(cv_img)
    img = (img - 127.5) / 127.5
    return img


def preprocess_depth_img(cv_img):
    cv_img = cv2.resize(cv_img, RESHAPE)
    img = np.array(cv_img)
    img = np.reshape(img, (RESHAPE[0], RESHAPE[1], 1))
    img = 2 * (img - 0.5)
    return img


if __name__ == "__main__":
    IMG_SRC = glob.glob("path/to/hazy/image/*.jpg")
    WEIGHT_SRC = glob.glob("./weights/g/*.h5")

    # txtfile = open("model_test_log.txt", "w")
    test_imgs = []
    label_imgs = []

    data_cnt = 0
    for img_path in IMG_SRC:
        img_name = get_file_name(img_path)

        sharp_img = cv2.imread(f"path/to/clean/image/{img_name}.jpg")
        sharp_img = cv2.cvtColor(sharp_img, cv2.COLOR_BGR2RGB)
        sharp_img = cv2.resize(sharp_img, (IMG_SIZE, IMG_SIZE))

        label_imgs.append(sharp_img)

        ori_image = cv2.imread(img_path)
        h, w, _ = ori_image.shape

        t = estimate_transmission(ori_image)
        t = preprocess_depth_img(t)

        ori_image = preprocess_cv2_image(ori_image)
        x_test = np.concatenate((ori_image, t), axis=2)
        x_test = np.reshape(x_test, (1, IMG_SIZE, IMG_SIZE, 4))
        test_imgs.append(x_test)
        data_cnt += 1
        print(f"Loaded {data_cnt} / {len(IMG_SRC)}")

    w_th = 0

    for weight_path in WEIGHT_SRC:
        txtfile = open("model_test_log.txt", "a+")

        model_name = get_file_name(weight_path)
        w_th += 1

        g = unet_spp_large_swish_generator_model()
        g.load_weights(weight_path)

        psnrs = []
        totaltime = 0

        cnt = 0

        for i in range(len(test_imgs)):
            x_test = test_imgs[i]
            sharp_img = label_imgs[i]

            start = timeit.default_timer()
            generated_images = g.predict(x=x_test)
            end = timeit.default_timer()
            infertime = end - start
            if cnt == 0:
                infertime = 0
            totaltime += float(infertime)

            de_test = deprocess_image(generated_images)
            de_test = np.reshape(de_test, (IMG_SIZE, IMG_SIZE, 3))
            # de_test = cv2.resize(de_test, (w, h))

            # rgb_de_test = cv2.cvtColor(de_test, cv2.COLOR_BGR2RGB)
            # cv2.imwrite(f"{output_dir}/{img_name}.jpg", rgb_de_test)

            psnr = calculate_psnr(de_test, sharp_img)

            psnrs.append(psnr)

            cnt += 1
            print(
                f"Weights: {w_th} / {len(WEIGHT_SRC)} - Images: {cnt} / {len(IMG_SRC)}"
            )

        average_psnr = np.mean(np.array(psnrs), axis=-1)
        average_time = totaltime / (len(IMG_SRC) - 1)

        # print("Average PSNR:", average_psnr)
        # print("Average Inference Time:", average_time)

        print(
            f"Model Name: {model_name}  PSNR: {average_psnr}  Time: {average_time}",
            file=txtfile,
        )

        # if w_th==1: break

        txtfile.close()

    print("Done!")
