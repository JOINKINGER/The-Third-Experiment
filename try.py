import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.font_manager import FontProperties
from skimage import img_as_float
from skimage.restoration import denoise_tv_chambolle
from scipy.ndimage import gaussian_filter
font = FontProperties(fname='C:/Windows/Fonts/simsun.ttc')  # 使用宋体字体
img = img_as_float(mpimg.imread('pic.jpg'))
# 添加噪声
noisy_img = img + 0.1 * img.std() * np.random.random(img.shape)
# 高斯模糊
gaussian_img = gaussian_filter(noisy_img, sigma=10)
# ROF 模型去噪
denoised_img = denoise_tv_chambolle(noisy_img, weight=0.1)
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(noisy_img, cmap='gray')
ax[0].set_title('原始噪声图像', fontproperties=font)
ax[1].imshow(gaussian_img, cmap='gray')
ax[1].set_title('高斯模糊图像 (σ=10)', fontproperties=font)
ax[2].imshow(denoised_img, cmap='gray')
ax[2].set_title('ROF 模型去噪图像', fontproperties=font)
for a in ax:
    a.axis('off')
plt.tight_layout()
plt.savefig('denoised_images.png')

noisy_img = img + 0.1 * img.std() * np.random.random(img.shape)
#向图像添加噪声。img.std() 计算图像的标准差，np.random.random(img.shape) 生成与图像形状相同的随机噪声，0.1 \* img.std() 调整噪声强度。
gaussian_img = gaussian_filter(noisy_img, sigma=10)
#使用 gaussian_filter 函数对噪声图像进行高斯模糊，sigma=10 指定高斯核的标准差。
denoised_img = denoise_tv_chambolle(noisy_img, weight=0.1)
#使用 denoise_tv_chambolle 函数对噪声图像进行 ROF 模型去噪，weight=0.1 指定去噪强度。