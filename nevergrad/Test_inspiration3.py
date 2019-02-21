
# coding: utf-8

# In[44]:


#---------------------- 1 useful declarations ----------------------
print('1: useful declarations')
import numpy as np
import torch
import copy
from torch.utils.serialization import load_lua
from torchvision.utils import make_grid,save_image
from torchvision.transforms import ToPILImage, ToTensor, Normalize, Resize
from IPython.display import display, clear_output
from PIL import Image
def load_image(img_path,show=False):
        img = Image.open(img_path).convert('RGB')
        if show:
            display(img.resize((512,512),Image.ANTIALIAS))
        return img
from torchvision.transforms import Scale
to_pil = ToPILImage()

#---------------------- 2 loads the generator network pgan ----------------------
print('2: load generator pgan')
import os
from torch.autograd import Variable
#from models.gan_visualizer import GANVisualizer
from models.progressive_gan import ProgressiveGAN
import json

#pathModel = '/private/home/oteytaud/datasets/yslcreativity4_s7_i96000.pt'
#pathConfig = '/private/home/oteytaud/datasets/yslcreativity4_train_config.json'

SZ = 128
if SZ == 256:
    sc = 6
elif SZ == 128:
    sc = 5
elif SZ == 512:
    sc = 7

#pathModel = '/private/home/oteytaud/datasets/PGAN_dtd/PGAN_dtd_s'+str(sc)+'_iter_48000.pt'
#pathConfig = '/private/home/oteytaud/datasets/PGAN_dtd/PGAN_dtd_s'+str(sc)+'_iter_48000_tmp_config.json'
pathModel = '/private/home/oteytaud/morgane/pytorch_GAN_zoo/PGAN_DTD20/default/default_s'+str(sc)+'_iter_48000.pt'
pathConfig = '/private/home/oteytaud/morgane/pytorch_GAN_zoo/PGAN_DTD20/default/default_s'+str(sc)+'_iter_48000_tmp_config.json'
with open(pathConfig, 'rb') as file:
    config = json.load(file)
    
print('we here load a pgan')
pgan  = ProgressiveGAN(useGPU= True, storeAvG = True, **config)
pgan.load(pathModel)



#---------------------- 3 Generate images from pgan ----------------------
print('3: generate images from pgan')
from torchvision.transforms import ToPILImage, ToTensor, Normalize, Resize
nimages= 10
noiseData, noiseLabels = pgan.buildNoiseData(nimages)
#texclass=1
#noiseLabels[0]=texclass # choosing the texture class
torch.save([noiseData,noiseLabels],'/private/home/oteytaud/HDGANSamples/random_gens/z.pth')
[noiseData,noiseLabels]= torch.load('/private/home/oteytaud/HDGANSamples/random_gens/z.pth')
#print(noiseData.shape)
img = pgan.test(noiseData, getAvG = True)

#for display in one row
total_width = SZ*nimages
max_height = SZ
new_im = Image.new('RGB', (total_width, max_height))  
x_offset = 0

print('we randomly generate ', nimages, ' images with the pgan')
for i in range(0,nimages):
    
    img2 = img[i].view(3,128,128).add(1).div(2) 
    img2 = img2.data.cpu()
    img2 = np.clip(img2, 0, 1)
    out = to_pil(img2)
    out.save('/private/home/oteytaud/HDGANSamples/random_gens/dtd_s'+str(sc)+'_rand_'+ str(i) +'.jpg') 
    #display(out)
    new_im.paste(out, (x_offset,0))
    x_offset += out.size[0]

new_im.save('/private/home/oteytaud/HDGANSamples/random_gens/dtd_s'+str(sc)+'_all_rand_' +'.jpg') 
print("display:",new_im)


# In[23]:


os.popen('ls ').read()


# In[63]:


import glob, os
 
os.chdir('/private/home/oteytaud/morgane/pytorch_GAN_zoo/')
dirpath = "/private/home/oteytaud/HDGANSamples/random_gens/"

import subprocess

nimages = 10

gs = 0.1
for rd in ["--gradient_descent ", "--random_search ", "--nevergradcma ", "--nevergradpso ", "--nevergradde ", "--nevergrad2pde ", "--nevergradpdopo ", "--nevergraddopo ", "--nevergradopo "]:
 nstep = 5000
 R = 0.1 # weight of the discriminator loss 
 L2 = 5  # weight of the rgb loss
 VGG = 1 # weight of the VGG loss 
 ind=0
 
 A = np.zeros((nimages))
 
 #for display in one row
 total_width = SZ*nimages
 max_height = SZ
 new_im = Image.new('RGB', (total_width, max_height))  
 x_offset = 0
 
 #for file in os.listdir('/private/home/oteytaud/datasets/dtd/images/blotchy'):
     #if file.endswith(".jpg"):
 for i in range(0,nimages):
     ind = ind + 1
     imgname = 'dtd_s'+str(sc)+'_rand_'+ str(i) 
     suffix = "inspiration_R_" +str(R)+"_VGG_"+ str(VGG) + "_L2_" + str(L2) + str(rd.split()[-1])+"nsteps"+str(nstep) 
     outname = dirpath+imgname+"_"+suffix+"/"+imgname+"_"+suffix+".jpg"
     
     im = load_image(dirpath + imgname +'.jpg') 
     print("target image in ", dirpath + imgname +'.jpg')
     #im = im.resize((SZ,SZ),Image.BICUBIC)
     #print("inspiration image")
     #display(im)
             
     cmd = "python eval.py inspirational_generation -m PGAN -n default -d PGAN_DTD10 -f /private/home/oteytaud/features_VGG19/VGG19_featureExtractor.pt id -s 5 -N 1 -R "+str(R)+" --weights "+ str(VGG) + " " + str(L2) +" --input_images "+dirpath+imgname+".jpg --np_vis -S "+suffix+" --nSteps "+ str(nstep)+" -l " + str(gs)+ " "+rd 
     proc = subprocess.Popen([cmd], stdout=subprocess.PIPE, shell=True)
     (out, err) = proc.communicate()
     print(out)
     #print(' we extract the learning rate')
     #print("out=", out)
  #   r_min = out[-12:]
   #  r_min = r_min[:-8]
     r_min = out.decode()  #.split(":")[-1]
     idx = r_min.index("rate :")
     r_min = r_min[(idx+6):].strip().split()[0].split("\n")[0]
     print("r_min=", float(r_min))
     A[i] = float(r_min)
     out = load_image(outname)
     print("rebuilt image in ", outname)
     #print("output result")
     #display(out)
     new_im.paste(out, (x_offset,0))
     x_offset += out.size[0]
 
 print("options:"+ suffix)
 # save reached optimal values    
 np.save(dirpath+imgname+"_"+suffix+'values', A)
 print(A)
 new_im.save(dirpath+imgname+"_"+suffix+'all.jpg') 
 display(new_im)
 


# In[24]:


imgname = "blotchy_0060"
suffix = "inspiration_r0"
dirpath = "/private/home/oteytaud/HDGANSamples/blotchy/"
outname = dirpath+imgname+"_"+suffix+"_0.jpg"
im = load_image(dirpath+imgname+".jpg")
im = im.resize((SZ,SZ),Image.BICUBIC)
print("inspiration image")
display(im)
os.system("python eval.py inspirational_generation -m PGAN -n PGAN_dtd -f /private/home/oteytaud/features_VGG19/VGG19_featureExtractor.pt id -s 5 -N 1 -R 0 --weights 1 100 --inputImage "+dirpath+imgname+".jpg --np_vis -S "+suffix+" --nSteps 500 --gs 0.01 --vrand 0")

out = load_image(outname)

print("output result")
display(out)


# In[5]:


imgname = "blotchy_0060"
suffix = "inspiration_r0"
dirpath = "/private/home/oteytaud/HDGANSamples/blotchy/"
outname = dirpath+imgname+"_"+suffix+"_0.jpg"
im = load_image(dirpath+imgname+".jpg")
im = im.resize((SZ,SZ),Image.BICUBIC)
print("inspiration image")
display(im)
os.system("python eval.py inspirational_generation -m PGAN -n PGAN_dtd -f /private/home/oteytaud/features_VGG19/VGG19_featureExtractor.pt id -s 5 -N 1 -R 1 --weights 1 100 --inputImage "+dirpath+imgname+".jpg --np_vis -S "+suffix+" --nSteps 500 --gs 0.01 --vrand 0")

out = load_image(outname)

print("output result")
display(out)


# In[6]:


imgname = "blotchy_0038"
suffix = "inspiration_r0"
dirpath = "/private/home/oteytaud/HDGANSamples/blotchy/"
outname = dirpath+imgname+"_"+suffix+"_0.jpg"
im = load_image(dirpath+imgname+".jpg")
im = im.resize((SZ,SZ),Image.BICUBIC)
print("inspiration image")
display(im)
os.system("python eval.py inspirational_generation -m PGAN -n PGAN_dtd -f /private/home/oteeytaud/features_VGG19/VGG19_featureExtractor.pt id -s 5 -N 1 -R 0 --weights 1 100 --inputImage "+dirpath+imgname+".jpg --np_vis -S "+suffix+" --nSteps 500 --gs 0.01 --vrand 0")

out = load_image(outname)

print("output result")
display(out)


# In[20]:


imgname = "blotchy_0060"
suffix = "inspiration_r0"
dirpath = "/private/home/oteytaud/HDGANSamples/blotchy/"
outname = dirpath+imgname+"_"+suffix+"_0.jpg"
im = load_image(dirpath+imgname+".jpg")
im = im.resize((SZ,SZ),Image.BICUBIC)
print("inspiration image")
display(im)
os.system("python eval.py inspirational_generation -m PGAN -n PGAN_dtd -f /private/home/oteytaud/features_VGG19/VGG19_featureExtractor.pt id -s 5 -N 1 -R 0 --weights 1 10 --inputImage "+dirpath+imgname+".jpg --np_vis -S "+suffix+" --nSteps 500 --gs 0.1 --vrand 0")

out = load_image(outname)

print("output result")
display(out)


# In[21]:


imgname = "blotchy_0038"
suffix = "inspiration_r0"
dirpath = "/private/home/oteytaud/HDGANSamples/blotchy/"
outname = dirpath+imgname+"_"+suffix+"_0.jpg"
im = load_image(dirpath+imgname+".jpg")
im = im.resize((SZ,SZ),Image.BICUBIC)
print("inspiration image")
display(im)
os.system("python eval.py inspirational_generation -m PGAN -n PGAN_dtd -f /private/home/oteytaud/features_VGG19/VGG19_featureExtractor.pt id -s 5 -N 1 -R 0 --weights 1 10 --inputImage "+dirpath+imgname+".jpg --np_vis -S "+suffix+" --nSteps 500 --gs 0.1 --vrand 1")

out = load_image(outname)

print("output result")
display(out)


# In[4]:


import glob, os
 
os.chdir('/private/home/oteytaud/SL_fashionGen')
dirpath = "/private/home/oteytaud/HDGANSamples/blotchy/"

gs = 0.1

rd = 1
nstep = 100
R = 0.1
L2 = 5
VGG = 1
ind=0
for file in os.listdir('/private/home/oteytaud/datasets/dtd/images/blotchy'):
    if file.endswith(".jpg"):
        #print(os.path.join("/mydir", file))

#for file in glob.glob("*.jpg"):
        imgname = file[:12]
        ind = ind + 1
        if ind < 2:
#    imgname = "blotchy_0040"
            suffix = "inspiration_R_" +str(R)+"_VGG_"+ str(VGG) + "_L2_" + str(L2) + "_rd_" + str(rd)
            outname = dirpath+imgname+"_"+suffix+"_0.jpg"
            im = load_image(dirpath+imgname+".jpg")
            im = im.resize((SZ,SZ),Image.BICUBIC)
            print("inspiration image")
            display(im)
            print
            os.system("python /private/home/oteytaud/SL_fashionGen/eval.py inspirational_generation -m PGAN -n PGAN_dtd -f /private/home/oteytaud/features_VGG19/VGG19_featureExtractor.pt id -s 5 -N 1 -R "+str(R)+" --weights "+ str(VGG) + " " + str(L2) +" --inputImage "+dirpath+imgname+".jpg --np_vis -S "+suffix+" --nSteps "+ str(nstep)+" --gs " + str(gs)+ " --vrand "+str(rd))
            print("python /private/home/oteytaud/SL_fashionGen/eval.py inspirational_generation -m PGAN -n PGAN_dtd -f /private/home/oteytaud/features_VGG19/VGG19_featureExtractor.pt id -s 5 -N 1 -R "+str(R)+" --weights "+ str(VGG) + " " + str(L2) +" --inputImage "+dirpath+imgname+".jpg --np_vis -S "+suffix+" --nSteps "+ str(nstep)+" --gs " + str(gs)+ " --vrand "+str(rd))
            print(outname)
            out = load_image(outname)

            print("output result")
            display(out)


# In[94]:


import glob, os
os.chdir('/private/home/oteytaud/datasets/dtd/images/blotchy')
for file in glob.glob("*.jpg"):
    print(file[:12])
    


# In[18]:


import numpy as np
import torch
import copy
from torch.utils.serialization import load_lua
from torchvision.utils import make_grid,save_image
from torchvision.transforms import ToPILImage, ToTensor, Normalize, Resize
from IPython.display import display, clear_output
from PIL import Image
def load_image(img_path,show=False):
        img = Image.open(img_path).convert('RGB')
        if show:
            display(img.resize((512,512),Image.ANTIALIAS))
        return img
from torchvision.transforms import Scale
to_pil = ToPILImage()

#---------------------- 2 loads the generator network pgan ----------------------
import os
from torch.autograd import Variable
#from models.gan_visualizer import GANVisualizer
from models.progressive_gan import ProgressiveGAN
import json


import glob, os
 
os.chdir('/private/home/oteytaud/SL_fashionGen')
dirpath = "/private/home/oteytaud/HDGANSamples/blotchy/"

gs = 0.05

rd = 1
nstep = 1000
R = 0.01
L2 = 150
VGG = 0.1
sc = 8

#for file in os.listdir('/private/home/oteytaud/datasets/dtd/images/blotchy'):
#    if file.endswith(".jpg"):
#        imgname = file[:12]
if rd >= 0:
    if rd >= 0:
        imgname = "blotchy_0044"
        suffix = "inspiration_PGANcreativity4_64000_R_" +str(R)+"_VGG_"+ str(VGG) + "_L2_" + str(L2) + "_rd_" + str(rd) + "_sc_"+str(sc)
        outname = dirpath+imgname+"_"+suffix+"_0.jpg"
        im = load_image(dirpath+imgname+"_crop_256.jpg")
        #im = im.resize((SZ,SZ),Image.BICUBIC)
        print("inspiration image")
        display(im)
        
        os.system("python /private/home/oteytaud/SL_fashionGen/eval.py inspirational_generation -m PGAN -n yslcreativity4 -f /private/home/oteytaud/features_VGG19/VGG19_featureExtractor.pt id -s "+str(sc)+" -N 1 -R "+str(R)+" --weights "+ str(VGG) + " " + str(L2) +" --inputImage "+dirpath+imgname+".jpg --np_vis -S "+suffix+" --nSteps "+ str(nstep)+" --gs " + str(gs)+ " --vrand "+str(rd)+" --crp 1" )

        print(outname)
        out = load_image(outname)
        #out = out.resize((256,256),Image.BICUBIC)
        print("output result")
        display(out)


# In[ ]:


import glob, os
 
os.chdir('/private/home/oteytaud/SL_fashionGen')
dirpath = "/private/home/oteytaud/HDGANSamples/blotchy/"
SZ =256
gs = 0.1

rd = 0
nstep = 500
R = 0.1
L2 = 5
VGG = 1
sc = 6

for file in os.listdir('/private/home/oteytaud/datasets/dtd/images/blotchy'):
    if file.endswith(".jpg"):
        #print(os.path.join("/mydir", file))

#for file in glob.glob("*.jpg"):
        imgname = file[:12]
#if rd >= 0:
#    imgname = "blotchy_0040"
        suffix = "inspiration_R_" +str(R)+"_VGG_"+ str(VGG) + "_L2_" + str(L2) + "_rd_" + str(rd) + "_sc_"+str(sc)
        outname = dirpath+imgname+"_"+suffix+"_0.jpg"
        im = load_image(dirpath+imgname+".jpg")
        im = im.resize((SZ,SZ),Image.BICUBIC)
        print("inspiration image")
        display(im)
        os.system("python /private/home/oteytaud/SL_fashionGen/eval.py inspirational_generation -m PGAN -n PGAN_dtd -f /private/home/oteytaud/features_VGG19/VGG19_featureExtractor.pt id -s "+str(sc)+" -N 1 -R "+str(R)+" --weights "+ str(VGG) + " " + str(L2) +" --inputImage "+dirpath+imgname+".jpg --np_vis -S "+suffix+" --nSteps "+ str(nstep)+" --gs " + str(gs)+ " --vrand "+str(rd))

        print(outname)
        out = load_image(outname)

        print("output result")
        display(out)


# In[ ]:


#python /private/home/oteytaud/SL_fashionGen/eval.py inspirational_generation -m PGAN -n yslcreativity4 -f /private/home/oteytaud/features_VGG19/VGG19_featureExtractor.pt id -s 7 -N 1 -R 0.1 --weights 1 5 --inputImage /private/home/oteytaud/HDGANSamples/blotchy/blotchy_0060.jpg --np_vis -S inspiration_crop --nSteps 500 --gs 0.1 --vrand 1 --crp 1

