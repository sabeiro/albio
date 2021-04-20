import os, sys, gzip, random, json, datetime, re, io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
import cv2
import io

dL = os.listdir(os.environ['HOME']+'/lav/src/')
sys.path = list(set(sys.path + [os.environ['HOME']+'/lav/src/'+x for x in dL]))
baseDir = os.environ['HOME'] + '/lav/viudi/graphic/logo_cnc/'

laserMax = 500
conf = {"x":200,"y":150,"min":0.1,"max":0.9,"res":3.5,"rotate":True,"fname":"soci2.jpg"}
conf['min'] = int(laserMax*conf['min'])
conf['max'] = int(laserMax*conf['max'])
delta = 1/conf['res']
imgF = baseDir + conf['fname']
img = mpimg.imread(imgF)
if conf["rotate"]: img = img.transpose(1,0,2)
aspect = img.shape[0]/img.shape[1]
conf['width'] = int(conf['x']*conf['res']) + 1
conf['height'] = int(conf['width']*aspect) + 1
conf['y'] = int(conf['height']/conf['res'])  
res = cv2.resize(img, dsize=(conf['width'], conf['height']), interpolation=cv2.INTER_CUBIC)
if False: # gradient
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            z = 4*j #+ 2*i
            res[i,j] = [z,z,z]
            
plt.imshow(res)
plt.show()

init = "$32=1\n(image to laser)\nG90\nG0 X0 Y0\nM3 S0\nF5000\n"
init = "G21 ; Set units to mm\n\
G90 ; Use absolute coordinates\n\
S0  ; Power off laser i.e. PWM=0\n\
M3  ; Activate Laser with dynamics\n\
F1500 ; Set speed\n"

cmdS = init
#cmdS += "G1 Z10\n"
x, y, s = 0, 0, 1
for i in range(res.shape[0]):
    i1 = int(y/conf['y']*conf['height'] - 0.0001)
    for j in range(res.shape[1]):
        x += s*delta
        if x > conf['x']: x = conf['x']; break
        if x < 0: x = 0; break
        j1 = int(x/conf['x']*conf['width'] - 0.0001)
        z = int(np.mean(res[i1,j1]))
        z = laserMax - int(z/255*laserMax)
        if z > conf['max']: z = conf['max']
        if z < conf['min']: continue
        st = "G1 X%.1f \nS%d\n" % (x,z)
        cmdS += st
    s *= -1
    y += delta
    if y > conf['y']: y = conf['y']; break
    x = conf['x'] if s < 0 else 0
    st = "G1 Y%.1f \nS%d\n" % (y,0)
    cmdS += st

cmdS += "M5\nG0 X0 Y0 Z0"

fName = imgF.split(".")[0] + ".nc"
with io.open(fName,"w",newline='\r\n') as f:
    f.write(cmdS)
    f.close()


