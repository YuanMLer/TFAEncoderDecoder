import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


__all__ = ['Attn']
class Attn:

    def __init__(self, th, tw):
        self.th = th
        self.tw = tw

    def mgnify(self, image:np.ndarray):
        h, w = image.shape
        th, tw = self.th, self.tw
        res = np.zeros((th, tw))
        rh = th//h + 1
        rw = tw//w + 1

        for i in range(th):
            for j in range(tw):
                res[i][j] = image[i//rh][j//rw]
        return res

    def save(self, image:np.ndarray, filename):
        m_image = self.mgnify(image)
        plt.imsave('tmp.png', m_image)
        grey_image = Image.open('tmp.png').convert('L')
        grey_image.save(filename)