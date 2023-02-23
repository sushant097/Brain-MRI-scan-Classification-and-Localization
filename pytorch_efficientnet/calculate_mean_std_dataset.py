import numpy as np
from PIL import Image
from pathlib import Path

dataset_dir = Path(f'./data/Training')
files = list(dataset_dir.rglob('*.jpg'))

print(len(files))

mean = np.array([0., 0., 0.])
stdTemp = np.array([0., 0., 0.])
std = np.array([0., 0., 0.])

numSamples = len(files)

for i in range(numSamples):
    im = Image.open(str(files[i])).convert("RGB")
    im = np.array(im).astype(np.float32) / 255.

    for j in range(3):
        mean[j] += np.mean(im[:, :, j])

mean = (mean / numSamples)

print(mean)

for i in range(numSamples):
    im = Image.open(str(files[i])).convert("RGB")
    im = np.array(im).astype(np.float32) / 255.
    for j in range(3):
        stdTemp[j] += ((im[:, :, j] - mean[j]) ** 2).sum() / (im.shape[0] * im.shape[1])

std = np.sqrt(stdTemp / numSamples)

print(std)

with open("mean_std.txt", 'w') as f:
    f.write(f"mean: {mean}\nstd: {std}")
