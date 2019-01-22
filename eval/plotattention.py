import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

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


attn_weights1 = np.load('atten1.npy')[:,0,:]
#print(attn_weights1)
print(attn_weights1.shape)
attn_weights2 = np.load('atten2.npy')[:,0,:]
#print(attn_weights2)
print(attn_weights2.shape)
# attn_weights = np.random.rand(16,16) + np.eye(16,16)*1.5
attn = Attn(800,800)
attn.save(attn_weights1, 'grey_attn1.png')
attn.save(attn_weights2, 'grey_attn2.png')




