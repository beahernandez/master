import glob
import numpy as np
import scipy.optimize as opt
from matplotlib import pyplot as plt
from numpy import random
from astropy.io import fits
import stomp

files=glob.glob('/vol/fohlen11/fohlen11_1/bhernandez/data/corr/udropouts/no_weight/Wtheta_W*zrgu_no_weight_RR_udropouts_m23.0t24.5')#magnitude_bins/individual_udropouts/23.2t23.4/Wtheta_W*zrgu_weight_RR_udropouts_m23.2t23.4')/vol/fohlen11/fohlen11_1/bhernandez/data/corr/udropouts/m23t24.5/new/Wtheta_W*zrgu_weight_RR_udropouts_m23t24.5')#Wtheta_*zrgu_no_weight_RR_udropouts_m23.0t24.5')#Wtheta_*zrgu_weight_RR_udropouts_m23t24')#folder+'/Wtheta*')
files.sort()
Wtheta=[]
theta=[]
total_rand=[]
 
  
covariance=np.zeros((len(theta),len(theta)))
corr=[]
randoms=[]
cat=np.loadtxt('galnumber_u_23t24.5.txt')
cat2=np.loadtxt('galnumber_name_u_23t24.5.txt',dtype='S16')
for itera in range(100):  #1000
    randrand=[]
    Wtheta=[]
    density=[]
    boot=random.randint(0, high=171, size=171)#189
    N=[]
    A=[]
    for i in range(20):
	RR_tmp=[]
	array=[]

	for f in boot:
	    fi=files[f]
	    data=np.loadtxt(fi)
	    name=fi[74:82]
	    if i==0:
		N.append(cat[f,0])
		A.append(cat[f,1])

	    array.append(data[i,1])
	    RR_tmp.append(data[i,-1])

	    
	Wtheta.append(np.nanmean(array))
	randrand.append(np.nanmean(RR_tmp))

	print i, itera
    
    Wtheta.append(np.sum(N)/np.sum(A))
    randrand.append(0)
    
    corr.append(Wtheta)#(np.array(DD)+np.array(RR)-2.*np.array(RD))/(np.array(RR)))

	
    randoms.append(randrand)


C=[]

theta=data[:,0]
theta=np.append(theta,0)
Cov=np.cov(corr,rowvar=0)
Wtheta=np.nanmean(corr, axis = 0)
errors=np.var(corr,axis=0)
RR=np.mean(randrand,axis=0)


final=[]
for i in range(len(theta)):
    final.append([theta[i],Wtheta[i],errors[i], randrand[i]])

   
np.savetxt('Wtheta_23.0t24.5_individual_no_weights_udropouts_density_test',final)
np.savetxt('Wcovar_23.0t24.5_individual_no_weights_udropouts_density_test',Cov)