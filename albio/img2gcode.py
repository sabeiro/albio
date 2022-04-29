import os, sys, gzip, random, json, datetime, re, io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
import cv2
import io

dL = os.listdir(os.environ['HOME']+'/lav/src/')
sys.path = list(set(sys.path + [os.environ['HOME']+'/lav/src/'+x for x in dL]))
baseDir = os.environ['HOME'] + '/lav/viudi/graphic/logo_cnc/'

conf = {"x":200,"y":150,"min":80,"max":500,"res":3.5,"rotate":False,"fname":"calib24.jpg","slide":10,"white":260}
#conf = {"x":200,"y":150,"min":80,"max":450,"res":3.5,"rotate":False,"fname":"comer1.jpg","slide":0,"white":230}
laserDelta = conf['max'] - conf['min']
delta = 1/conf['res']
imgF = baseDir + conf['fname']
img = mpimg.imread(imgF)
if img.shape[0] > img.shape[1]: img = img.transpose(1,0,2)
aspect = img.shape[0]/img.shape[1]
conf['width'] = int(conf['x']*conf['res']) + 1 
conf['height'] = int(conf['width']*aspect) + 1
conf['y'] = int(conf['height']/conf['res'])  
res = cv2.resize(img, dsize=(conf['width'], conf['height']), interpolation=cv2.INTER_CUBIC)
if False: # gradient
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            z = np.mean(res[i,j])
            z = 0#4*j #+ 2*i
            res[i,j] = [z,z,z]
            
# plt.imshow(res)
# plt.show()
prev = np.ones((res.shape[0],res.shape[1],res.shape[2]))*255

init = "$32=1\n(image to laser)\nG90\nG0 X0 Y0\nM3 S0\nF5000\n"
init = "G21 ; Set units to mm\n\
G90 ; Use absolute coordinates\n\
S0  ; Power off laser i.e. PWM=0\n\
M3  ; Activate Laser with dynamics\n\
F1500 ; Set speed\n"


cmdS = init
x, y, z, x_prev, y_prev, z_prev = 0, 0, 0, 0, 0, 0
s, zl, xi = 1, -1000, 0
lineL = []
for i in range(res.shape[0]):
    y = i/res.shape[0]*conf['y']
    iL = np.array((range(res.shape[1])))
    if s < 0: iL = res.shape[1] - 1 - iL
    for j in iL:
        x = conf['x'] - j/res.shape[1]*conf['x']
        z = int(np.mean(res[i,j]))
        if z > conf['white']: continue
        zl = conf['max'] - int(z/255*laserDelta)
        if zl < conf['min']: continue
        prev[i,j] = [z,z,z]
        if abs(z - z_prev) < conf['slide']:
            prev[i,j] = [z_prev,z_prev,z_prev]
            continue
        x_prev, y_prev, z_prev = x, y, z
        #x += (.5 - np.random.uniform())/conf['res']*.5
        
        st = "G1 X%.2f \nS%d\n" % (x,zl)
        cmdS += st
        xi += 1

    # if s < 0: st = "\nS%d\nG1 X%.2f " % (0,0.)
    # else : st = "S%d\nG1 X%.2f \n" % (0,conf['x'])
    if xi == 0: continue
    xi = 0
    s *= -1
    st = "S0\nG1 Y%.2f \nS%d\n" % (y,0)
    cmdS += st
    z_prev = -1000

cmdS += "G1 X0 Y0 Z0\nM5\nG0 X0 Y0 Z0\n"
lineL = pd.DataFrame(lineL)

fName = imgF.split(".")[0] + ".nc"
with io.open(fName,"w",newline='\r\n') as f:
    f.write(cmdS)
    f.close()

plt.imshow(prev.astype(int))
plt.show()


