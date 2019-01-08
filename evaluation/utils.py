import sys
import os
import matplotlib.pylab  as plt

sys.path.append("..")
from shapenet import get_split

def get_ply(split_file='data/splits.json'):
    js = get_split(split_file)
    return js

def plot_images(rows,
                cols,
                images,
                filename,
                color=True,
                dir_name="figures"):

    os.makedirs(dir_name, exist_ok=True)
    assert(len(images) == (rows * cols))

    fig = plt.figure(figsize=(cols * 2, rows * 2))
    image_size = images[0].shape[0]
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        image = images[i]
        if color:
            plt.imshow(image)
        else:
            image = np.reshape(image, [image_size, image_size])
            plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    path = os.path.join(dir_name, filename)
    plt.savefig(path)

