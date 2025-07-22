import glob

import cv2
from skimage.metrics import structural_similarity as cal_ssim
from core.utils import get_file_name

total_psnr = 0
total_ssmi = 0

TEST_SET = glob.glob("./ohaze_valset_gt/*.jpg")

OUTPUT_FOLDER = "predict_ohaze_unet_spp_swish_deeper_gan_4c"
TXT_FILE = open(f"./{OUTPUT_FOLDER}/test_log.txt", "w+")

for path in TEST_SET:
    fname = get_file_name(path)

    pred = cv2.imread(f"./{OUTPUT_FOLDER}/{fname}.jpg")

    if pred is None:
        print(f"Failed to load image {OUTPUT_FOLDER}/{fname}.jpg")
        continue

    gt = cv2.imread(path)

    psnr = cv2.PSNR(pred, gt)
    ssmi = cal_ssim(gt, pred, data_range=pred.max() - pred.min(),
                    multichannel=True)

    print(fname, psnr, ssmi)
    print(fname, psnr, ssmi, file=TXT_FILE)

    total_psnr += psnr
    total_ssmi += ssmi

average_psnr = total_psnr / len(TEST_SET)
average_ssmi = total_ssmi / len(TEST_SET)

print("PSNR:", average_psnr)
print("SSMI:", average_ssmi)

print("PSNR:", average_psnr, file=TXT_FILE)
print("SSMI:", average_ssmi, file=TXT_FILE)
