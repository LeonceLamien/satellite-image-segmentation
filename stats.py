import cv2
import numpy as np
import glob

files = glob.glob("Dataset/images/*.png")  # adapte si besoin

means = []
stds = []

for f in files:
    img = cv2.imread(f)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    means.append(img.mean(axis=(0,1)))
    stds.append(img.std(axis=(0,1)))

mean = np.mean(means, axis=0)
std = np.mean(stds, axis=0)

print("MEAN =", mean)
print("STD  =", std)
