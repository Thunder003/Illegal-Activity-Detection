#!/usr/bin/env python
# coding: utf-8

# In[1]:



get_ipython().system('pip install scikit-image')


# In[1]:


import numpy as np
import cv2
from skimage.measure import compare_ssim
import argparse
from PIL import Image
import imutils
import copy


# In[72]:


img1=cv2.imread('E:\code\M.L\StartUp\AerialForestBeforeAfter\im1.jpg')
img2=cv2.imread('E:\code\M.L\StartUp\AerialForestBeforeAfter\im2.jpg')
img1=cv2.resize(img1,(400,500))
img2=cv2.resize(img2,(400,500))
print(img1.shape)
print(img2.shape)
cv2.imshow('image1',img1)
cv2.imshow('image2',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[12]:


height=500
width=400

img1_1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) 
img2_1 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) 

orb_detector = cv2.ORB_create(5000) 
kp1, d1 = orb_detector.detectAndCompute(img1_1, None) 
kp2, d2 = orb_detector.detectAndCompute(img2_1, None) 
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True) 
matches = matcher.match(d1, d2) 
matches.sort(key = lambda x: x.distance) 

# Take the top 90 % matches forward. 
matches = matches[:int(len(matches)*0.9)] 
no_of_matches = len(matches) 
  
# Define empty matrices of shape no_of_matches * 2. 
p1 = np.zeros((no_of_matches, 2)) 
p2 = np.zeros((no_of_matches, 2)) 
  
for i in range(len(matches)): 
  p1[i, :] = kp1[matches[i].queryIdx].pt 
  p2[i, :] = kp2[matches[i].trainIdx].pt 
  
 #Find the homography matrix 
homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC) 
  
# Use this matrix to transform the 
# colored image wrt the reference image. 
img2 = cv2.warpPerspective(img2, 
                    homography, (width, height)) 


# In[4]:


cv2.imshow('image1',img1)
cv2.imshow('image2',img2)
cv2.waitKey(300)
cv2.destroyAllWindows()


# In[73]:


mask=np.zeros(img1.shape, dtype=np.uint8)
roi= np.array([[(20,0), (400,0), (400,350),(0,350)]], dtype=np.int32)
ignore=(255,255,255)
cv2.fillPoly(mask, roi, ignore)
m1 = cv2.bitwise_and(img1, mask)
m2 = cv2.bitwise_and(img2, mask)


# In[92]:


cv2.imshow('image1',m1)
cv2.imshow('image2',m2)
cv2.waitKey(000)
cv2.destroyAllWindows()


# In[93]:


# m1=cv2.pyrMeanShiftFiltering(m1,5,5)
m1=cv2.GaussianBlur(m1,(3,3),cv2.BORDER_DEFAULT)
# m2=cv2.pyrMeanShiftFiltering(m2,5,5)
m2=cv2.GaussianBlur(m2,(3,3),cv2.BORDER_DEFAULT)


# In[94]:


cv2.imshow('image1',m1)
cv2.imshow('image2',m2)
cv2.waitKey(300)
cv2.destroyAllWindows()


# In[103]:


grayA = cv2.cvtColor(m1, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(m2, cv2.COLOR_BGR2GRAY)
grayA=cv2.equalizeHist(grayA)
grayB=cv2.equalizeHist(grayB)
def creatcopyimg(m1,m2):
    im1x=copy.deepcopy(m1) 
    im2x=copy.deepcopy(m2)
    return im1x,im2x
im1x,im2x=creatcopyimg(m1,m2)


# In[104]:


cv2.imshow('image1',im1x)
cv2.imshow('image2',im2x)
cv2.waitKey(300)
cv2.destroyAllWindows()


# In[105]:


def coun(diff,x1,y1,coimg1,coimg2,w1,h1,bo):
    thresh=cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 50) 
#     _,thresh = cv2.threshold(diff,250,255,cv2.THRESH_BINARY)
#     thresh = cv2.threshold(diff, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
#     thresh = cv2.dilate(thresh, np.ones((3,3),np.uint8),iterations=1)
#     kernel = cv2.getStructuringElement(thresh,cv2.MORPH_OPEN,(3,3))
#     thresh = cv2.dilate(thresh, kernel)
    if(bo<0):
        cv2.imshow("diff", diff)
        cv2.imshow("Mothre", thresh)
        #     cv2.imshow("coimg1", coimg1)
        #     cv2.imshow("coimg2", coimg2)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
    cnts= cv2.findContours(thresh.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        (x, y , w, h) = cv2.boundingRect(c)
        G=cv2.contourArea(c)
#         print(x,y,w,h,G)
        if(w*h>200 and x!=0 and y!=0):
            cv2.rectangle(im1x, (x + x1, y + y1), (x + x1 + w, y + y1 + h), (0, 0, 255), 1)
            cv2.rectangle(im2x, (x + x1, y + y1), (x + x1 + w, y + y1 + h), (0, 0, 255), 1)
    return       


# In[106]:


def compareblock(imga,imgb,w,h):
    imgstore=[]
    ime=imga
    thre=0;
    bo=0
    count=0;
    it=0;
    for y in range(0,350,50):
        for x in range(0,imga.shape[1]-w,50):
            coimg1=imga[y:y+h,x:x+w]
            coimg2=imgb[y:y+h,x:x+w]
#             coimg1=cv2.morphologyEx(coimg1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations = 10)
#             coimg2=cv2.morphologyEx(coimg2, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations = 10)
            (score, diff) = compare_ssim(coimg1, coimg2, full=True)
            thre=thre+score
            diff = (diff * 255).astype("uint8")
            if(score<2):
                coun(diff,x,y,coimg1,coimg2,w,h,bo)
                bo=bo+1
            imgstore.insert(count,diff)
            count=count+1
            
#                 if(count==1475):
#                     cv2.imshow('image',diff)
#                     print(diff)
#                     cv2.waitKey(0)
#                     cv2.destroyAllWindows()
                
            it=it+1       
#             print(count,it,x,y)
    return imgstore,thre/count                


# In[120]:


height, width = grayB.shape[:2] 
value=[]  
ls=0
graybcopy=copy.deepcopy(grayB)        
for i in range(0,15,1):
    im1x,im2x=creatcopyimg(m1,m2)
    
    quarter_height, quarter_width = 0, -width / (100-i)
    T = np.float32([[1, 0, quarter_width], [0, 1, quarter_height]]) 
    graybcopy = cv2.warpAffine(graybcopy, T, (width, height))
    
    ls=ls+quarter_width
    changeimg=copy.deepcopy(grayA)
    mask=np.zeros(grayA.shape, dtype=np.uint8)
    roi= np.array([[(20,0), (400+ls,0), (400+ls,350),(0,350)]], dtype=np.int32)
    ignore=(255,255)
    cv2.fillPoly(mask, roi, ignore)
    changeimg = cv2.bitwise_and(changeimg, mask)
    
#     print(grayB.shape)
#     print(grayA.shape)
#     cv2.imshow("grayA", grayA)
#     cv2.imshow("grayB", graybcopy)
#     cv2.imshow("Changeimg", changeimg)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
    li,thre=compareblock(changeimg,graybcopy,250,250)
    value.insert(0,(thre,ls))


# In[121]:


value.sort(reverse=True, key=lambda x:x[0])

print(value)


# In[129]:


im1x,im2x=creatcopyimg(m1,m2)

shift=value[0][1]

graybcopy=copy.deepcopy(grayB)
T = np.float32([[1, 0, shift], [0, 1, quarter_height]]) 
graybcopy = cv2.warpAffine(graybcopy, T, (width, height))
    
changeimg=copy.deepcopy(grayA)
mask=np.zeros(grayA.shape, dtype=np.uint8)
roi= np.array([[(20,0), (400+shift,0), (400+shift,350),(0,350)]], dtype=np.int32)
ignore=(255,255)
cv2.fillPoly(mask, roi, ignore)
changeimg = cv2.bitwise_and(changeimg, mask)

cv2.imshow("graybcopy", graybcopy)
cv2.imshow("changeimg", changeimg)
cv2.waitKey(0)
cv2.destroyAllWindows()

li,thre=compareblock(changeimg,graybcopy,250,250)


# In[128]:


cv2.imshow("Original", im1x)
cv2.imshow("Modified", im2x)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[503]:


thresh = cv2.threshold(diff, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)


# In[20]:


for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    print(x,y,w,h)
    cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), 2)


# In[92]:


cv2.imshow("Original", img1)
cv2.imshow("Modified", img2)
cv2.imshow("diff", diff)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


grayA = cv2.GaussianBlur(grayA,(,9),600,cv2.BORDER_DEFAULT)
grayB = cv2.GaussianBlur(grayB,(9,9),600,cv2.BORDER_DEFAULT)

diff= cv2.bilateralFilter(diff, 15,75 , 75) 

ab=img1[:,0:200,:]
cv2.imshow('image',ab)
cv2.waitKey(0)
cv2.destroyAllWindows()

(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))


# In[23]:


c=2
q=1

print(c,q)


# In[135]:





# In[136]:





# In[137]:





# In[ ]:




