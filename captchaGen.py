import os
import logging
import random

import captcha
import numpy as np
import cv2
from captcha.image import ImageCaptcha


os.environ['DISPLAY'] = ":0"
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s %(name)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


logger = logging.getLogger("I'm {}".format(str(__file__).split(os.sep)[-1]))
logger.setLevel(logging.DEBUG)

numChars = 4
inputShape = (96, 320)
n = 4
w = 10
charsInCaptcha = ["0", "1", "2", "4" ,"5"]

ABIs = []

f_data = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')


for i in range(numChars):
    i += 1
    ABI = np.zeros(shape=(inputShape[0], n*w))
    ABI[:, :] = 0
    cB = (i-1)*w + 1
    cF = i * w
    ABI[ : , cB:cF] = 255
    ABIs.append(ABI)


# Remove all data
os.system("rm -rf {}/*".format(f_data))
# Create folder
for c in charsInCaptcha:
    os.system('mkdir -p {}/{}'.format(f_data, str(c)))

# Gen data
imgStore = ImageCaptcha(width=inputShape[1], height=inputShape[0])
for i in range(100):
    chars = random.sample(charsInCaptcha, k=4)
    content = ''.join(chars)

    img = imgStore.generate(content)
    img = np.asarray(bytearray(img.read()), dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)

    for idx, c in enumerate(chars):
        captc = np.hstack((ABIs[idx], img))
        cv2.imshow("Image", img)
        cv2.imshow("captc", captc)
        f = os.path.join(f_data, str(c), "{}.png".format(i))
        cv2.imwrite(f, captc)
        logger.info(f)
