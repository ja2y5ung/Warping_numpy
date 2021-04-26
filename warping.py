import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from numpy.linalg import inv
import math

img = Image.open("img.jpg").convert('L')
img = np.array(img)

img2 = Image.open("img2.jpg").convert('L')
img2 = np.array(img2)

global war_zeros
war_zeros = np.zeros_like(img)

def zicsun(x1,y1,x2,y2):
    global X,Y
    x = np.array(range(600))  
    y = np.array(range(800))
    X,Y = np.meshgrid(x,y)
    z = ((y2 - y1)/(x2 - x1))*(X - x1)+(y1 - Y)
    giul = (y2 - y1)/(x2 - x1)
    return z, giul

def dot_choice(image,image2):
    plt.imshow(image,'gray')
    print ( "Please click" )
    dot1 = (np.around(plt.ginput(20)).T).astype(np.int)
    print(dot1)
    plt.imshow(image2,'gray')
    print ( "Please click" )
    dot2 = (np.around(plt.ginput(20)).T).astype(np.int)
    print(dot2)
    
    return dot1, dot2

def area_choice(dot):
    z1, g1 = zicsun(dot[0][0],dot[1][0],dot[0][1],dot[1][1])
    z2, g2 = zicsun(dot[0][0],dot[1][0],dot[0][2],dot[1][2])
    z3, g3 = zicsun(dot[0][3],dot[1][3],dot[0][2],dot[1][2])
    z4, g4 = zicsun(dot[0][3],dot[1][3],dot[0][1],dot[1][1])
    if g2<0 and g4<0:
        vv = np.where((z1<0) & (z2<0) & (z3>0) & (z4>0))

        return vv
    elif g2<0:
        vv = np.where((z1<0) & (z2<0) & (z3>0) & (z4<0))
        return vv
    elif g4<0:
        vv = np.where((z1<0) & (z2>0) & (z3>0) & (z4>0))
        return vv
    else:
        vv = np.where((z1<0) & (z2>0) & (z3>0) & (z4<0))
        return vv

def warping(dot1,dot2,image_vec):
    x = dot1[0]
    y = dot1[1]
    x_y = x*y
    n1 = np.ones((1,np.size(x)))
    xy = np.vstack((x_y,x,y,n1))
    xy_1 = np.linalg.pinv(xy)
    change = np.dot(dot2,xy_1)

    xx = image_vec[1]
    yy = image_vec[0]
    xx_yy = xx*yy
    n2 = np.ones((1,np.size(xx)))
    xxyy = np.vstack((xx_yy,xx,yy,n2))

    war_picture_vec = np.int_(np.dot(change,xxyy))
    war_zeros[war_picture_vec[1],war_picture_vec[0]] = img[image_vec[0],image_vec[1]]

    return war_zeros

#################### main #########################
    
img_d,img2_d = dot_choice(img,img2)

for i in range(0,15):
    if i+1 % 3 == 0 and i != 0:
            continue
    img_d_l = np.array([[img_d[0][i], img_d[0][i+1], img_d[0][i+4], img_d[0][i+5]],
                        [img_d[1][i], img_d[1][i+1], img_d[1][i+4], img_d[1][i+5]]])
    
    img2_d_l = np.array([[img2_d[0][i], img2_d[0][i+1], img2_d[0][i+4], img2_d[0][i+5]],
                        [img2_d[1][i], img2_d[1][i+1], img2_d[1][i+4], img2_d[1][i+5]]])

    choice_vec1 = np.array(area_choice(img_d_l))                         
    zeros = warping(img_d_l,img2_d_l,choice_vec1)

plt.imshow(zeros,'gray')
plt.show()








