import numpy as np
import cv2

def clust_rgb(image,k=5,iters=5): 
    img=image.copy()
    h,w,c=img.shape
    orig=image.copy()
    Klusters=np.random.randint(256,size=(k,3))
    for it in range(iters):
        img=image.copy()
        for i in range(h):
            for j in range(w):
                pix=img[i][j]
                diff=np.sqrt(np.sum((Klusters-pix)**2,axis=1))
                c=np.argmin(diff)
                img[i][j]=Klusters[c]
        l=[]
        for i in range(k):
            Ys,Xs,c=np.where(img==Klusters[i])
            kth_points=orig[Ys,Xs]
            l.append(np.sum(Klusters[i]-kth_points))
            Klusters[i]=np.mean(kth_points,axis=0)
        
    return img

def clust_gray(gray,k=3,iters=3): 
    
    img=gray.copy()
    h,w=img.shape
    orig=gray.copy()
    Klusters=np.random.randint(256,size=k)
    print('init clusters', Klusters)
    for it in range(iters):
        img=gray.copy()
        for i in range(h):
            for j in range(w):
                pix=img[i][j]
                diff=np.abs(Klusters-pix)
                c=np.argmin(diff)
                img[i][j]=Klusters[c]
        l=[]
        for i in range(k):
            Ys,Xs=np.where(img==Klusters[i])
            kth_points=orig[Ys,Xs]
            l.append(np.sum(Klusters[i]-kth_points))
            Klusters[i]=np.mean(kth_points)
    return img
'''
oimage = cv2.imread("seg1.jpg")#,cv2.IMREAD_GRAYSCALE)

print(oimage.shape[-1])
#image = cv2.cvtColor( oimage , cv2.COLOR_RGB2Luv)

#gray = cv2.cvtColor(oimage,cv2.COLOR_BGR2GRAY)

new_img = clust_gray(oimage)

cv2.imshow('image' , oimage)
cv2.imshow('output' , new_img )
cv2.waitKey(0)
cv2.destroyAllWindows()
'''