import glob
import numpy as np
import scipy.optimize as opt
from matplotlib import pyplot as plt
from numpy import random
import os.path


#def chisqfunc((a,b,c)):
    #chisq=0
    #model=a*data[0]**b -c
    #C=np.linalg.inv(covariance)
    #for i in range(len(data[0])-1):
	##modeli = 10**(a + b*np.log10(np.float(data[0][i])))
	##print modeli, data[1][i], np.float(data[0][i]), a, b
	#for j in range(len(data[0])-1):
	    ##modelj = 10**(a + b*np.log10(np.float(data[0][j])))
	    #chisq = chisq + ((np.float(data[1][i]) - model[i])*C[i,j]*(np.float(data[1][j]) - model[j]))
    
    #return chisq


#Interactive input to select folder
#folder=input('Which folder?')
#Looks for every file with correlations in that folder
files=glob.glob('/vol/fohlen11/fohlen11_1/bhernandez/data/corr/magnitude_bins/individual_udropouts/24.2t24.4/Wtheta_W*zrgu_weight_RR_udropouts_m24.2t24.4')#/vol/fohlen11/fohlen11_1/bhernandez/data/work/Wtheta_W*zrgu_no_weight_RR_gdropouts_m24.9t25.1')#corr/magnitude_bins/udropouts_0.2/weight_maps/Wtheta_W*zrgu_weight_RR_udropouts_m23t23.2')#/users/bhernandez/thesis/work/Wtheta_W*zrgu_no_weight_RR_udropouts_m23t23.2')#zrgu_weight_RR_udropouts_m23t23.2')#folder+'/Wtheta*')

#Initiates the variables
Wtheta=[]
theta=[]
total_rand=[]
idx=0

#Opens each file and appends all the correlations separated by the different thetas
#for f in files:
  #tmp=[]
  #fi = open(f, 'r')
  #randoms=[]
  #for line in fi:
    #line = line.strip()
    #columns = line.split()
    #randoms.append(np.float(columns[-1]))
    #tmp.append(np.float(columns[1]))
    #if idx==0:
      #theta.append(np.float(columns[0]))
  #idx=1
  #Wtheta.append(tmp)
  #total_rand.append(randoms)
  
  
covariance=np.zeros((len(theta),len(theta)))
corr=[]
randoms=[]
cat=np.loadtxt('galnumber_u_24.2t24.4.txt')
cat2=np.loadtxt('galname_u_24.2t24.4.txt',dtype='S16')
for itera in range(1000):  
    boot=random.randint(0, high=171, size=171)
    randrand=[]
    #randgal=[]
    #galgal=[]
    c=[]
    N=[]
    A=[]
    for i in range(20):
	RR_tmp=0
	RD_tmp=0
	DD_tmp=0
	for f in boot:   
	    #if os.path.exists(files[f]+'new'):
		#name=files[f]+'new'
	    #else:
	    name=files[f]
	    if i==0:
		N.append(cat[f,0])
		A.append(cat[f,1])

	    data=np.loadtxt(name)
	    RR_tmp+=data[i,-1]
	    RD_tmp+=data[i,-2]
	    DD_tmp+=data[i,-3]
	    #c_tmp.append(data[i,1])
	c.append((RR_tmp+DD_tmp-2*RD_tmp)/(RR_tmp))
	randrand.append(RR_tmp)
	#randgal.append(RD_tmp)
	#galgal.append(DD_tmp)
	#print RR_tmp, DD_tmp,RD_tmp
    c.append(np.sum(N)/np.sum(A))
    corr.append(c)
    print itera
#corr.append((np.array(DD)*np.array(RR))/(np.power(np.array(RD),2))-1)
    
    #randoms.append(data[:,-1])
	
theta=data[:,0]
Cov=np.cov(corr,rowvar=0)
Wtheta=np.nanmean(corr, axis = 0)
errors=np.var(corr,axis=0)
RR=np.mean(randrand,axis=0)
final=[]
for i in range(len(theta)):
    final.append([theta[i],Wtheta[i],errors[i], randrand[i]])


np.savetxt('Wtheta_24.2t24.4_udropouts_weight_density',final)
np.savetxt('Wcovar_24.2t24.4_udropouts_weight_density',Cov)
#C=[]

#final=[]
#Wtheta=[]
#for k in range(len(corr[0])):
    #Wtheta_tmp=[]
    #for i in range(len(corr)):
	##w=[]
	##for j in range(len(corr)):
	    ##w.append(corr[j][i][k])
	    ###print corr[j][i][k]
	#Wtheta_tmp.append(corr[i][k])
    ##print Wtheta_tmp
    #print k
    #Wtheta.append(np.mean(Wtheta_tmp))

#final.append([theta,np.mean(Wtheta),np.var(Wtheta)/np.sqrt(len(Wtheta)*1.)])


#Calculates covariance matrix

#for i in range(len(data[:,0])):
    #for j in range(len(data[:,0])):
	#covariance[i,j]+=((corr[i,:]-np.mean(corr[:,])*(np.float(Wtheta[k][j])-f[j][1]))
		
#C.append(covariance/((len(Wtheta))*1.*(len(Wtheta)*1.-1.))) #
	
##mask=np.logical_or(np.isnan(C),np.isinf(C))
##tmp=np.array(C)
##C_mask=tmp[mask]
#for i in range(20):
    #for j in range(20):
	#mask=np.invert(np.logical_or(np.isnan(C[:][i][j]),np.isinf(C[:][i][j])))
	#tmp=np.array(C[:][i][j])
	#C_mask=tmp[mask]
	#covariance[i,j]=np.mean(C_mask)
##covariance=np.mean(C)
##Wtheta_tmp=np.zeros((len(Wtheta),20))
##for j in range(len(Wtheta)):
    ##for i in range(len(Wtheta[0])):
	
	##Wtheta_tmp[j,i]=Wtheta[j][i]
#Cov=np.cov(Wtheta,rowvar=0)
  
  
##prints out the result to a file
#print f
#out=open('/vol/fohlen11/fohlen11_1/bhernandez/data/corr/udropouts/small_mean_variance_weights_small_scales_RR.txt','w')
#for h in range(len(f)):
    #out.write("%s %f %f %f\n" % (theta, np.mean(corr), np.var(corr),np.mean(randoms)]))

#out.close()

##Saves covariance and minimizes chi squared
#np.savetxt('/vol/fohlen11/fohlen11_1/bhernandez/data/corr/udropouts/covariance_small_scales_RR.txt', covariance, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ')
#g=np.array(f)
#data=[g[0:9,0],g[0:9,1]]#[[np.log10(np.float(y)) for y in g[0:9,0]],[np.log10(np.float(y)) for y in g[0:9,1]]]
#x0 = np.array([1,0.9,10**(-4)])
##result =  opt.fmin(chisqfunc, x0)#, method = 'Powell')#'Nelder-Mead')

###fit_param=result#.values()[5]

###Plotting the results
##dif='/vol/fohlen11/fohlen11_1/bhernandez/data/corr/gdropouts/small_mean_variance_weights_small_scales.txt'
##data2=np.loadtxt('/users/bhernandez/thesis/omega_ij_err_clean_20_FFV_IC_Masim.dat')
##plt.figure()
##i=0
##f=dif
##if f==dif:#for f in dif:
  ###Separates the variables
  ##fi = open(f, 'r')
  ##i=i+1
  ##theta=[]
  ##Wtheta=[]
  ##err=[]
  ##for line in fi:
    ##line = line.strip()
    ##columns = line.split()
    ##if float(columns[1])>0:
      ##theta.append(float(columns[0])*60.)
      ##Wtheta.append(float(columns[1]))
      ##err.append(float(columns[2]))#**(1/2))
  ##name=f[55:61]
  ###Plotting
  ##plt.errorbar(theta,Wtheta,yerr=err,fmt='x',label=name)
  ###w = polyfit(np.log10(theta[0:10]),np.log10(Wtheta[0:10]),1)
  ###fit = fit_param[0]*data[0]**fit_param[1] -fit_param[2]#10**(fit_param[0]*np.log10(theta)+fit_param[0])
  ##plt.errorbar(data2[:,0],data2[:,1],data2[:,2],fmt='k--', label='Hendriks data')
  ##plt.xscale('log')
  ##plt.yscale('log')
  ##plt.xlabel('theta [arcmin]')
  ###plt.xlim(0.1,100)
  ###plt.ylim(0.001,10)
  ##plt.ylabel('w_theta')
  ##plt.title('Correlation function of the individual pointings')
  ##plt.legend(numpoints=1)

##plt.savefig('dif_magnitudes.png')
##plt.show()

