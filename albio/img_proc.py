import os, sys, gzip, random, json, datetime, re, io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
import cv2
import io

dL = os.listdir(os.environ['HOME']+'/lav/src/')
sys.path = list(set(sys.path + [os.environ['HOME']+'/lav/src/'+x for x in dL]))
baseDir = os.environ['HOME'] + '/lav/aff/'

conf = {"x":20,"y":150,"min":25,"max":200,"res":3,"rotate":True}
laserMax = 255
delta = 1/conf['res']
imgF = baseDir + "raw/img.jpg"
img = mpimg.imread(imgF)
if conf["rotate"]: img = img.transpose(1,0,2)
aspect = img.shape[0]/img.shape[1]
conf['width'] = int(conf['x']*conf['res']) + 1
conf['height'] = int(conf['width']*aspect) 
res = cv2.resize(img, dsize=(conf['width'], conf['height']), interpolation=cv2.INTER_CUBIC)
plt.imshow(res)
plt.show()

cmdS = "$32=1\n(image to laser)\nG90\nG0 X0 Y0\nM3 S0\nF5000\n"
x, y, s = 0, 0, 1
for i in range(res.shape[0]):
    i1 = int(y/conf['y']*conf['height'])
    for j in range(res.shape[1]):
        x += s*delta
        if x > conf['x']: x = conf['x']; break
        if x < 0: x = 0; break
        j1 = int(x/conf['x']*conf['width'])
        z = int(np.mean(res[i1,j1]))
        if z > conf['max']: continue
        if z < conf['min']: z = conf['min']
        z = laserMax - int(z/255*laserMax)
        st = "G1 X%.1f\nS%d\n" % (x,z)
        cmdS += st
    y += delta
    if y > conf['y']: y = conf['y']; break
    x = x
    st = "G1 Y%.1f \nS%d\n" % (y,0)
    cmdS += st
    s *= -1

cmdS += "M5\nG0 X0 Y0"

with io.open(imgF + ".nc", "w",newline='\r\n') as f:
    f.write(cmdS)
    f.close()


