import numpy as np
import scipy.optimize as opt
import pylab as plt
import emcee
import scipy.integrate as inte
from scipy import interpolate
from scipy.interpolate import interp1d
import glob
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from scipy import optimize
from scipy import special
from scipy import spatial
import corner
import os
import subprocess
import os.path
import correlation
import stomp
import cosmology
#import halo_v2 as halo
import halo
import hod
import kernel
import mass_function
import matplotlib
import math
#######Configuring matplotlib##############################
matplotlib.rcParams['xtick.major.size']= 6
matplotlib.rcParams['ytick.major.size']= 6
matplotlib.rcParams['ytick.minor.size']= 4
matplotlib.rcParams['xtick.minor.size']= 4
matplotlib.rcParams['xtick.major.width']= 1.
matplotlib.rcParams['ytick.major.width']= 1.
matplotlib.rcParams['ytick.minor.width']= 1.
matplotlib.rcParams['xtick.minor.width']= 1.
matplotlib.rcParams['axes.linewidth']= 2.0
###########################################################

def IC_det(w):

    N=281.211640212#15482.4603175 #u dropouts
    #N=398.899470899#3809.83597884 #g dropouts
    #print w
    C=(1.+np.array(w))*N**(-2.)*np.sum(w)
    return C

def IC_integrand(lnkx,lnky,lnkz):
    
    #lx=1.*deg_to_rad
    #ly=1.*deg_to_rad
    #lz=0.5
    #if kx<0.0000001 or ky<0.0000001:
	#print 'Wk=0'
	#Wk=0.
    #else:
	#print kx,ky,kz
    kx=np.exp(lnkx)
    ky=np.exp(lnky)
    kz=np.exp(lnkz)
    kxlx=kx*lx/2.
    kyly=ky*ly/2.
    kzlz2=np.power(kz*lz,2)/2.
    Wk=np.abs(np.power((np.exp(-kzlz2)*np.sin(kxlx)/(kxlx)*np.sin(kyly)/(kyly)),2))
    Pk=P(np.sqrt(np.power(kx,2)+np.power(ky,2)+np.power(kz,2)))
    #print kx,ky,kz
    return Pk*Wk*(kx*ky*kz)

def galaxy_density(filename,V):
    """
    Computes the galaxy density and the error for the information in the imput file. 
    This file should have the number galaxy per pointing. The comoving volume must
    also be specified.
        
    Args:
	filename: file where the number of galaxies is found 
	V: comoving volume of each pointing
    
    Returns:
	Galaxy density and error
    
    """
    
    galdens=np.loadtxt(filename)
    gal_num=np.mean(galdens)#48321.
    err=np.var(galdens)/(len(galdens))**(1/2.)
    gal_dens=gal_num/V
    dens_err=err/V
    
    return gal_dens,dens_err

def comoving_volume(z,d_omega,sigma_z):
    """
    This function computes the comoving volume in the standard cosmology for a 
    given redshift, a solid angle and a Gaussian redshift distribution
        
    Args:
	z: mean redshift 
	d_omega: solid angle in radians
	sigma_z: dispersion of the Gaussian redshift distribution
    
    Returns:
	Comoving volume
    
    """
    #d_z=dndz*#inte.romberg(lambda x:1./(sigma_z*np.sqrt(2*np.pi)*
				  #np.exp((x-z)**2/(2.*sigma_z**2))), 0., 5.)
    
    #Ez=cosmo_multi.E(z)
    Om=0.3 - 4.15e-5/0.7**2
    c=3.*10**8
    Dh=1./cosmo_multi.H0
    #print Dh
    #DA=cosmo_multi.angular_diameter_distance(z)
    Vc=inte.romberg(lambda x: (Dh*(1.+x)**2 * 
		    cosmo_multi.angular_diameter_distance(x)**2/cosmo_multi.E(x)*
		    d_omega*1.)*pz_high(x), 0, 4)
		    #*pz_high(x)
    return Vc

#def lnprior(A):#(M_min, M_1,alpha)):
    #"""
    #Computes the prior for the fitting.
        
    #Args:
	#Parameters: Tuple containing the parameters used for the fitting
    
    #Returns:
	#Infinity if the criteria is not satisfied and 0 if it is
    
    #"""
    ##if -5.0 < M_min < 0.5 and 0.0 < M_1 < 10.0:
    ##if -100<A<100 and -20<delt<20: #10.<M_min<15. and 10.<M_1<15. and 0.<alpha<2.: #and IC>=0.0 and 0.<alpha<2.:
        ##return 0.0
    ##return -np.inf
    #return 0.0

#def lnprob(A,data,covariance):#(M_min,M_1,alpha),  data,covariance):
    #lp = lnprior(A)#(M_min, M_1,alpha))
    #if not np.isfinite(lp):
        #return -np.inf
    #return lp - 0.5*chisqrfunc(A,  data, covariance )#(M_min,M_1,alpha),  data, covariance )

def function_corr(logtheta, corr):#Wtheta_H):
    theta=np.exp(logtheta)
    dlogtheta=1
    dtheta=theta*dlogtheta
    return  corr(logtheta)*dtheta


def function_theta(logtheta):
    return logtheta#10**(logtheta)#

def integrand_N_noexp(M,M_1,alpha):
    Ng=(M/M_1)**alpha
    return Ng*mass(np.log(M))/M

def integrand_N(M,M_1,alpha):
    Ng=(M/M_1)**alpha
    return Ng*mass(np.log(M))/M

def integrand_N_occnum(M,M_1,alpha):
    #M in log
    Ng=(np.exp(M)/M_1)**alpha
    return Ng*mass(M)*np.exp(M)#Ng*mass(np.log(M))/M #Ng*mass.f_nu(mass.nu(M))

def integrand_Nden(M):
    return mass(M)*np.exp(M)#mass(np.log(M))/M

def integrand_M(M,M_1,alpha):
    Ng=(np.exp(M)/M_1)**alpha
    return Ng*mass(M)*np.exp(M)**2#Ng*mass(np.log(M))

def integrand_Mden(M,M_1,alpha):
    Ng=(np.exp(M)/M_1)**alpha
    return Ng*mass(M)*np.exp(M)#Ng*mass(np.log(M))/M

def func_fit(x, B, A): # this is your 'straight line' y=f(x)
    return -B*x + A

def func_fit_2((A, B),x,y): # this is your 'straight line' y=f(x)
    #B=gamma-1
    #print gamma,r0
    #A=r0**gamma*(inte.romberg(lambda z:pz_high(z)**2*
			  #((1+z)*cosmo_multi.angular_diameter_distance(z))**(1-gamma)*
			  #cosmo_single.E(z)**-1,0,5)*
	          #special.beta(1/2.,(gamma-1)/2.))/(inte.romberg(pz_high,0, 5))**2
	    
    return np.abs(y-A*x**(-B))

def func_fit_3(x,A, B): # this is your 'straight line' y=f(x)
    #B=gamma-1
    #print gamma,r0
    #A=r0**gamma*(inte.romberg(lambda z:pz_high(z)**2*
			  #((1+z)*cosmo_multi.angular_diameter_distance(z))**(1-gamma)*
			  #cosmo_single.E(z)**-1,0,5)*
	          #special.beta(1/2.,(gamma-1)/2.))/(inte.romberg(pz_high,0, 5))**2
	          
    #print A,B,x
	    
    return A*x**(-B)

def chisqrfunc(d1h,d2h,data,C,M_min,M_1,alpha):#(M_min,M_1,alpha),data,C):
    #name=f[8:]

    #parameters=name.split("_")
    #M_min=np.float(parameters[0])
    #M_1=np.float(parameters[1])
    #alpha=np.float(parameters[2])
    chisq=0
    Wtheta_H=(d1h[:-1,1]+d2h[:-1,1])
    theta=(d1h[:-1,0])
    logtheta=np.log(data[:,0])#theta)#
    correlations_o = interpolate.InterpolatedUnivariateSpline(theta, Wtheta_H)
    Wtheta=correlations_o(data[:,0]) #corr_low.correlation(data[:,0]*deg_to_rad)+correlations_o(data[:,0]) #0.12486203902693749**2*corr_low.correlation(data[:,0]*deg_to_rad)+0.87513796097306273**2*
    correlations = interpolate.InterpolatedUnivariateSpline(logtheta, Wtheta)

    RR=data[:,-1]
    av=[]
    av_test=[]
    num=0
    idx=0
    angle=[]
    
    for theta_bin in wtheta_ang.Bins():
	if 14>num>=5:
	    logTMin=np.log(theta_bin.ThetaMin())
	    logTMax=np.log(theta_bin.ThetaMax())
	    TMin=theta_bin.ThetaMin()
	    TMax=theta_bin.ThetaMax()
	    idx+=1
	    #print num,len(av)
	    angle.append((TMin+TMax)/2.)
	    w=inte.romberg(function_corr,logTMin,logTMax,args=[correlations],vec_func=True)/(TMax-TMin)

	    av.append(w)#np.abs(inte.romberg(function_theta,logTMin,logTMax)))
	    
	num+=1

    
    N=inte.romberg(integrand_N,M_min,8.9*10**14,args=[M_1,alpha])*V#*(180./np.pi)**2
    Wtheta=data[:,1]
    #c=3.*10**8
    #P=np.fft.fft(W_dm[:,1])
    #Wtheta=np.append(Wtheta,density)
    
    #RR=data[:,-1]
    #IC=(np.sum(RR*np.array(av))/np.sum(RR))*np.ones_like(Wtheta)
    #IC[-1]=0
    #10**
    

    IC=IC_det(av)
    IC_ar=[]
    for i in range(len(IC)):
	IC_ar.append(IC[i])
    IC_ar.append(0)
    #IC_ar=IC*np.ones_like(Wtheta)
    #IC_ar[-1]=0
    #/inte.romberg(integrand_Nden, M_min,10**16)
    #M_halo=inte.romberg(integrand_M,M_min,10**16,args=[M_1,alpha])/inte.romberg(integrand_Mden, M_min,10**16,args=[M_1,alpha])
    A=117.42
    av.append(N)#N/A)
    #print N
    
    for i in range(len(Wtheta)):
	for j in range(len(Wtheta)):
	    #if np.isfinite(Wtheta[j]) and np.isfinite(Wtheta[i]):
	    chisq += (Wtheta[i] - av[i] + IC_ar[i])*C[i,j]*(Wtheta[j] - av[j] + IC_ar[j]) #+IC av_test[j]
	    #print chisq
		
    #print Wtheta
    #print chisq,N
    return chisq

#def compute_errors(M_min,M_min_incr,M_1,M_1_incr,alpha,alph_incr,MIN,plus=bool):
    #incrM_min=0
    #incrM_1=0
    #incralpha=0
    #changed=False
    #name='%s_%s_%s'%(M_min,M_1,alpha)
    #chi_dif=0
    #while chi_dif<1:
	#proc=[]
	#if plus==True:
	    #incrM_min+=M_min_incr*M_min
	    #incrM_1+=M_1_incr*M_1 
	    #incralpha+=alph_incr*alpha
	#else:
	    #incrM_min-=M_min_incr*M_min
	    #incrM_1-=M_1_incr*M_1
	    #incralpha-=alph_incr*alpha
	##print incr
	#print incrM_min,incrM_1,incralpha
	#if not os.path.exists('corr_1h_%s_%s_%s'%(M_min+incrM_min,M_1+incrM_1,alpha+incralpha)):
	    #arg1="nice -n15 ./acf_1h_v2.exe -f corr_1h_%s_%s_%s -q QZ/z_PofZ_Udropouts_modified_v3.dat -mn %s -m1 %s -al %s"%(M_min+incrM_min,M_1+incrM_1,alpha+incralpha,M_min+incrM_min,M_1+incrM_1,alpha+incralpha)
	    #arg2="nice -n15 ./acf_2h_v2.exe -f corr_2h_%s_%s_%s -q QZ/z_PofZ_Udropouts_modified_v3.dat -mn %s -m1 %s -al %s"%(M_min+incrM_min,M_1+incrM_1,alpha+incralpha,M_min+incrM_min,M_1+incrM_1,alpha+incralpha)
	    #proc.append(subprocess.Popen(arg1,shell='True'))
	    #proc.append(subprocess.Popen(arg2,shell='True'))

	    #proc[0].wait()
	    #proc[1].wait()
	#d1h=np.loadtxt('corr_1h_%s_%s_%s'%(M_min+incrM_min,M_1+incrM_1,alpha+incralpha))
	#d2h=np.loadtxt('corr_2h_%s_%s_%s'%(M_min+incrM_min,M_1+incrM_1,alpha+incralpha))
	#chisqtmp=chisqrfunc(d1h,d2h,data,C)
	#chi_dif=chisqtmp-MIN
	#print chi_dif
	#if chi_dif<0:
	    #M_min+=incrM_min
	    #M_1+=incrM_1
	    #alpha+=incralpha
	    #MIN=chisqtmp
	    #name='%s_%s_%s'%(M_min,M_1,alpha)
	    #incrM_min=0
	    #incrM_1=0
	    #incralpha=0
	    #changed=True
	    #print 'changed'

    #incr=max(abs(incrM_min),abs(incrM_1),abs(incralpha))
    #return incr,changed,MIN,M_min,M_1,alpha,name


def lnprior((M_min,M_1,alpha)):
    #lnIC = c
    #print 'hey', a,b,c
    if 10**10 < M_min < 10**14 and  10**10< M_1 < 10**14 and 0.0 < alpha < 2.0:
        return 0.0
    return -np.inf

def lnprob((logM_min,logM_1,alpha),neighbors):
    M_min=10**(logM_min)
    M_1=10**(logM_1)
    lp = lnprior((M_min,M_1,alpha))
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike((M_min,M_1,alpha),neighbors)

def lnlike((M_min,M_1,alpha),neighbors):
    
    chisq=0

    weights,values=neighbors.query([M_min,M_1,alpha], k=3)
    #M_min1,M_11,alpha1=param[values[0]]
    #M_min2,M_12,alpha2=param[values[1]]
    #M_min3,M_13,alpha3=param[values[2]]
    
    name1=paramnames[values[0]]
    name2=paramnames[values[1]]
    name3=paramnames[values[2]]
    
    parameters=name1.split("_")
    M_min1=np.float(parameters[0])
    M_11=np.float(parameters[1])
    alpha1=np.float(parameters[2])
    
    parameters=name2.split("_")
    M_min2=np.float(parameters[0])
    M_12=np.float(parameters[1])
    alpha2=np.float(parameters[2])
    
    parameters=name3.split("_")
    M_min3=np.float(parameters[0])
    M_13=np.float(parameters[1])
    alpha3=np.float(parameters[2])
    #name1='%s_%s_%s'%(M_min1,M_11,alpha1)
    #name2='%s_%s_%s'%(M_min2,M_12,alpha2)
    #name3='%s_%s_%s'%(M_min3,M_13,alpha3)
    
    
    d1h1=np.loadtxt('corr_1h_'+name1)
    d1h2=np.loadtxt('corr_1h_'+name2)
    d1h3=np.loadtxt('corr_1h_'+name3)
    
    d2h1=np.loadtxt('corr_2h_'+name1)
    d2h2=np.loadtxt('corr_2h_'+name2)
    d2h3=np.loadtxt('corr_2h_'+name3)
    
    wt=1./weights[0]+1./weights[1]+1./weights[2]
    
    chisq=chisqrfunc(d1h1,d2h1,data,C,M_min1,M_11,alpha1)*(1./(weights[0]*wt))+chisqrfunc(d1h2,d2h2,data,C,M_min2,M_12,alpha2)*(1./(weights[1]*wt))+chisqrfunc(d1h3,d2h3,data,C,M_min3,M_13,alpha3)*(1./(weights[2]*wt))
    corr=((d1h1[:,1]+d2h1[:,1])*(1./(weights[0]*wt))+(d1h2[:,1]+d2h2[:,1])*(1./(weights[1]*wt))+(d1h3[:,1]+d2h3[:,1])*(1./(weights[2]*wt)))
    N_tmp=inte.romberg(integrand_N_occnum,np.log(M_min),np.log(8.9*10**14),args=[M_1,alpha])/inte.romberg(integrand_Nden, np.log(M_min),np.log(8.9*10**14))
    M_tmp=inte.romberg(integrand_M,np.log(M_min),np.log(8.9*10**14),args=[M_1,alpha])/inte.romberg(integrand_Mden, np.log(M_min),np.log(8.9*10**14),args=[M_1,alpha])
    
    Nfile.write("%s "%N_tmp)
    Mfile.write("%s "%M_tmp)
    corrfile.write("%s \n"%corr)

    print M_min,M_1,alpha,chisq
    return -chisq



#def chisqfunc((A, delt)):#, lnIC):
    #chisq=0
    #model=A*data[0]**delt -ICdet(A*data[0]**delt)#np.log(IC)
    #C=np.linalg.inv(covariance)
    #for i in range(len(data[0])-1):
	#for j in range(len(data[0])-1):
	    #chisq = chisq + ((np.float(data[1][i]) - model[i])*C[i,j]*(np.float(data[1][j]) - model[j]))
    #print chisq
    #return chisq

def mcmc_fitting(initial_guess, chain_length,neighbors):
    """
    Performs the mcmc fitting.
        
    Args:
	initial_guess: Tuple containing the initial parameters with which
			the chain will start.
	data: the data which is going to be fitted
	chain_length: The chain length. How many times will the 
			parameters be computed. This will also be the number 
			of points in parameter space.
    
    Returns:
	The result of the mcmc fitting with errors.
    
    """
    ndim=len(initial_guess)
    nwalkers = chain_length#(0.1*initial_guess) #(0.1*np.array(initial_guess)) #1e-4
    pos = [np.array(initial_guess) + (0.1*np.array(initial_guess)*np.random.randn(ndim)) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[neighbors]) #threads=9,

    print("Running MCMC...")
    sampler.run_mcmc(pos, 500, rstate0=np.random.get_state())
    print("Done.")

    burnin = 50
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

    #samples[:, 2] = np.exp(samples[:, 2])
    result = map(lambda v: 
		(v[1], v[2]-v[1], v[1]-v[0]),
		zip(*np.percentile(samples, [16, 50, 84],
		axis=0)))
		
        
    return result, sampler


#open('N_23t24.dat','w+').close()
#open('Mhalo_23t24.dat','w+').close()
#open('correlations_23t24.dat','w+').close()

#gal_dens, dens_err=galaxy_density('/users/bhernandez/thesis/work/galdens_u.txt',V)
deg_to_rad = np.pi/180.0
p_z=np.loadtxt('QZ/pz_low.dat')#/vol/fohlen11/fohlen11_1/bhernandez/data/p_z/z_PofZ_Udropouts.dat')
pz_tmp=np.loadtxt('QZ/z_PofZ_Udropouts.dat')
A=inte.simps(pz_tmp[:,1],x=pz_tmp[:,0])
pz_high=interpolate.InterpolatedUnivariateSpline(pz_tmp[:,0], pz_tmp[:,1]/A)
#max_pz=np.max(p_z)
#p_z=p_z/max_pz
cosmo_dict = {
    "omega_m0": 0.3 - 4.15e-5/0.7**2,
    "omega_b0": 0.046,
    "omega_l0": 0.7,
    "omega_r0": 4.15e-5/0.7**2,
    "cmb_temp": 2.726,
    "h"       : 0.7,
    "sigma_8" : 0.800,
    "n_scalar": 0.960,
    "w0"      : -1.0,
    "wa"      : 0.0
    }

cosmo_single_low = cosmology.SingleEpoch(redshift=0.1, cosmo_dict=cosmo_dict,
				with_bao=False)
cosmo_single= cosmology.SingleEpoch(redshift=3.0, cosmo_dict=cosmo_dict,
				with_bao=False)
cosmo_multi = cosmology.MultiEpoch(z_min=0.0, z_max=5.0, cosmo_dict=cosmo_dict,
			    with_bao=False)
halo_dict = {
    "stq": 0.3,
    "st_little_a": 0.707,
    "c0": 9.,
    "beta": -0.13,
    "alpha": -1,
    "delta_v": -1
    }

			    
				 
mass_file_low='/vol/fohlen11/fohlen11_1/bhernandez/chomp/mVector_PLANCK-SMT_z0.1.txt'
mass_low = mass_function.MassFunctionExternal(mass_file_low,redshift=0.1, 
				    cosmo_single_epoch=cosmo_single_low,
				    halo_dict=halo_dict)
				    
mass_file='/vol/fohlen11/fohlen11_1/bhernandez/chomp/mVector_PLANCK-SMT z: 3.0.txt'
#mass = mass_function.MassFunctionExternal(mass_file,redshift=3.0, 
				    #cosmo_single_epoch=cosmo_single,
				    #halo_dict=halo_dict)
				    
mass_tmp=np.loadtxt(mass_file)
mass=interpolate.InterpolatedUnivariateSpline(np.log(mass_tmp[:,0]),mass_tmp[:,6])
#mass = mass_function.MassFunction(redshift=3.0, 
				    #cosmo_single_epoch=cosmo_single,
				    #halo_dict=halo_dict)
hod_dict_ini = {"log_M_min":12.14,
	"sigma":     0.15,
	"log_M_0":  12.14,
	"log_M_1p": 13.43,
	"alpha":      1.0,
	"w":	      1.0}


sdss_hod = hod.HODZheng(hod_dict_ini)

halo_model_low= halo.Halo(redshift=0.1, input_hod=sdss_hod,
		    cosmo_single_epoch=cosmo_single_low, mass_func=mass_low)


lens_dist_low = kernel.dNdzInterpolation(p_z[:25,0],p_z[:25,1])#dNdzGaussian(0.0, 5.0, 3.1, 0.05)#0.1)#0.5


lens_window_low = kernel.WindowFunctionGalaxy(lens_dist_low, cosmo_multi)
con_kernel_low = kernel.Kernel(ktheta_min=0.001*0.001*deg_to_rad,
			ktheta_max=100.0*1.0*deg_to_rad,
			window_function_a=lens_window_low,
			window_function_b=lens_window_low,
			cosmo_multi_epoch=cosmo_multi)

V=comoving_volume(3.,0.0356402381150449,0.5)#0.01745,0.5)
gal_dens, dens_err=galaxy_density('/users/bhernandez/thesis/work/galdens_u.txt',V)

corr_low = correlation.Correlation(theta_min_deg=0.01,
			    theta_max_deg=2.0,
			    input_kernel=con_kernel_low,
			    input_halo=halo_model_low)#,
			    #power_spec='power_gg')
corr_low.compute_correlation()

data=np.loadtxt('/users/bhernandez/thesis/work/Wtheta_24.2t24.4_udropouts_weight_density')#'/vol/fohlen11/fohlen11_1/bhernandez/data/corr/udropouts/final/Wtheta_23t24_individual_weights')#'/vol/fohlen11/fohlen11_1/bhernandez/data/corr/udropouts/final/Wtheta_udropouts_m23t24_with_proper_weight')#'/vol/fohlen11/fohlen11_1/bhernandez/data/corr/magnitude_bins/udropouts_0.2/Wtheta_combined_24.2t24.4_udropouts') #'/users/bhernandez/thesis/work/Wtheta_23t24_udropouts')#'/vol/fohlen11/fohlen11_1/bhernandez/data/corr/udropouts/final/Wtheta_pointings_weights_small_scales_m23t24_RR.txt')#Wtheta_combined_small_scales_noregions_m23t24')
#data=np.log10(data)
density=data[-1,1]
data=data[5:-7,:]
Wtheta=data[:,1]
theta=data[:,0]
covariance_tmp=np.loadtxt('/users/bhernandez/thesis/work/Wcovar_24.2t24.4_udropouts_weight_density')#'/vol/fohlen11/fohlen11_1/bhernandez/data/corr/udropouts/final/Wcovar_23t24_individual_weights')#/users/bhernandez/thesis/work/Wcovar_23t24.5_combined')#'/vol/fohlen11/fohlen11_1/bhernandez/data/corr/udropouts/final/Wcovar_udropouts_m23t24_with_proper_weight')#'/vol/fohlen11/fohlen11_1/bhernandez/data/corr/magnitude_bins/udropouts_0.2/Wcovar_combined_24.2t24.4_udropouts') #'/users/bhernandez/thesis/work/covariance_23t24_udropouts')#'/vol/fohlen11/fohlen11_1/bhernandez/data/corr/udropouts/final/Wcovar_pointings_weights_small_scales_m23t24_RR.txt')#Wtheta_combined_small_scales_noregions_m23t24')
#covariance=covariance_tmp.flatten()[np.isfinite(covariance_tmp.flatten())]
#covariance=covariance.reshape((np.sqrt(covariance.shape[0]),np.sqrt(covariance.shape[0])))

cov_dens_row=covariance_tmp[-1,5:-7]
cov_dens_col=np.append(covariance_tmp[5:-7,-1],covariance_tmp[-1,-1])
covariance=covariance_tmp[5:-7,5:-7]
covariance=np.insert(covariance,9, cov_dens_row,axis=0)
covariance=np.insert(covariance,9,cov_dens_col,axis=1)
c=(2.998*10**5)
C=np.linalg.inv(covariance)
#print np.dot(C,covariance)
#print covariance

wtheta_ang = stomp.AngularCorrelation(0.00016666666666666666, 100.0/60.0, 5.)
##################GRID###################
chisq=[]
param=[]
files=glob.glob('corr_1h_*')
lz=cosmo_multi.comoving_distance(3.25)-cosmo_multi.comoving_distance(2.75) #cosmo_multi.comoving_distance(3.)
#3000.*inte.romberg(lambda x:(1./cosmo_multi.E(x)),2.75,3.25)
lx=0.8*deg_to_rad*cosmo_multi.comoving_distance(3.)
ly=lx
P_tmp=np.loadtxt('/users/bhernandez/Downloads/kVector_PLANCK-SMT.txt')
P=interpolate.InterpolatedUnivariateSpline(P_tmp[:,0],P_tmp[:,1])
#I=inte.tplquad(IC_integrand,-3,2,lambda x:-3,lambda x:2,lambda x,y:-3,lambda x,y:2)
I=[0.008540213770896643, 7.714259139243831e-09]#[0.00188666891970308, 4.4348340759816675e-09]#I=[54.86737333985418,7.595535082936543e-07] #[47.12101568443874, 1.6912579605647738e-07]
IC_tmp=8./(2.*np.pi)**3*I[0]
dif=100.
IC=0.
#(61.064104495188595, 7.342517362169105e-07)
#(61.064104495188595, 7.342517362169105e-07)
files_IC=glob.glob('/vol/fohlen11/fohlen11_1/bhernandez/data/corr/magnitude_bins/individual_udropouts/24.2t24.4/Wtheta_W*zrgu_weight_RR_udropouts_m24.2t24.4')
data_full=data
sigma=[]
DD=[]
RR=[]
while dif>0.001:
    lin=data[:,1]+IC#np.log10(data[4:8,1]+IC) #
    linang=data[:,0]*60.##np.log10(data[4:8,0]) #
    #lin=np.log10(av[4:8])
    #linang=np.log10(angle[4:8]
    c=(2.998*10**5) #km/s
    H0=100. #km/s/Mpc/h
    if len(linang)>1:
	A,beta=curve_fit(func_fit_3, linang, lin)[0]
	print A, beta
	#x,s = optimize.leastsq(func_fit_2, np.array([0.1,0.8]), args=(linang, lin))#(func_fit_2, linang, lin,np.array([0.9,0.01]))[0]
	#A,beta=x
	gamma=beta+1
	r0_int=(A*(inte.romberg(pz_high,2, 5))**2/ #FINE
		(inte.romberg(lambda z:pz_high(z)**2* #FINE
			    ((1+z)*cosmo_multi.angular_diameter_distance(z)*(deg_to_rad)/60.)**(1-gamma)*
			    (c/H0*np.sqrt(1/cosmo_single.E0(z)))**-1,2,5)*
		special.beta(1/2.,(gamma-1)/2.))) #FINE
		    #r0_int=(A*(inte.romberg(pz_high,0, 5))**2/
		#(inte.romberg(lambda z:pz_high(z)**2*
			    #((1+z)*cosmo_multi.angular_diameter_distance(z))**(1-gamma)*
			    #cosmo_single.E(z)**-1,0,5)*
		#special.beta(1/2.,(gamma-1)/2.))) 
		#c*cosmo_multi.E(z) *(deg_to_rad)
	r0=r0_int**(1/gamma)
	#*(1./deg_to_rad)
	sigma8g=72.*(r0/8)**gamma/((3-gamma)*(4-gamma)*(6-gamma)*2**gamma)
	sigma8z=(0.834*cosmo_multi.growth_factor(3.))**2#cosmo_multi.sigma_m(8.0, redshift=3)**2#cosmo_multi.luminosity_distance(3.)*0.9#cosmo_multi.sigma_m(8.0, redshift=3)
	b2=sigma8g/sigma8z
	if np.isfinite(r0):
	    sigma=b2*IC_tmp
    
    for f in files_IC:
	data_IC=np.loadtxt(f)
	data_tmp=data_IC[5:15,:] #4:15
	#w=data_tmp[:,1]
	#a=[idx for idx in range(len(w)) if 500>w[idx] >0.009]
	#lin=data_tmp[a,1]+IC#np.log10(data[4:8,1]+IC) #
	#linang=data_tmp[a,0]*60.##np.log10(data[4:8,0]) #
	##lin=np.log10(av[4:8])
	##linang=np.log10(angle[4:8]
	#c=(2.998*10**5) #km/s
	#H0=100. #km/s/Mpc/h
	#if len(linang)>1:
	    #A,beta=curve_fit(func_fit_3, linang, lin)[0]
	    #print A, beta
	    ##x,s = optimize.leastsq(func_fit_2, np.array([0.1,0.8]), args=(linang, lin))#(func_fit_2, linang, lin,np.array([0.9,0.01]))[0]
	    ##A,beta=x
	    #gamma=beta+1
	    #r0_int=(A*(inte.romberg(pz_high,2, 5))**2/ #FINE
		    #(inte.romberg(lambda z:pz_high(z)**2* #FINE
				#((1+z)*cosmo_multi.angular_diameter_distance(z)*(deg_to_rad)/60.)**(1-gamma)*
				#(c/H0*np.sqrt(1/cosmo_single.E0(z)))**-1,2,5)*
		    #special.beta(1/2.,(gamma-1)/2.))) #FINE
			##r0_int=(A*(inte.romberg(pz_high,0, 5))**2/
		    ##(inte.romberg(lambda z:pz_high(z)**2*
				##((1+z)*cosmo_multi.angular_diameter_distance(z))**(1-gamma)*
				##cosmo_single.E(z)**-1,0,5)*
		    ##special.beta(1/2.,(gamma-1)/2.))) 
		    ##c*cosmo_multi.E(z) *(deg_to_rad)
	    #r0=r0_int**(1/gamma)
	    ##*(1./deg_to_rad)
	    #sigma8g=72.*(r0/8)**gamma/((3-gamma)*(4-gamma)*(6-gamma)*2**gamma)
	    #sigma8z=(0.834*cosmo_multi.growth_factor(3.))**2#cosmo_multi.sigma_m(8.0, redshift=3)**2#cosmo_multi.luminosity_distance(3.)*0.9#cosmo_multi.sigma_m(8.0, redshift=3)
	    #b2=sigma8g/sigma8z
	    #if np.isfinite(r0):
		#sigma.append(b2*IC_tmp)
	DD.append(np.nanmean(data_IC[:,3]))
	RR.append(np.nanmean(data_IC[:,-1]))
	
    IC_old=IC
    IC=1/np.sum(RR)*sigma*np.sum(np.array(DD))
	
    dif=0.
    for i in range(len(data_full[:8,1])):
	dif+=np.sqrt(((data_full[i,1]+IC_old)-(data_full[i,1]+IC))**2)
    print IC,dif

 
M_min_array=[]
M_1_array=[]
alpha_array=[]
param=[]
paramnames=[]
files.sort()
chisq=[]
for f in files:
    d1h=np.loadtxt(f)
    name=f[8:]
    parameters=name.split("_")
    M_min=np.float(parameters[0])
    M_1=np.float(parameters[1])
    alpha=np.float(parameters[2])
    d2h=np.loadtxt('corr_2h_'+name)
    param.append([M_min,M_1,alpha])
    paramnames.append(name)
    M_min_array.append(M_min)
    M_1_array.append(M_1)
    alpha_array.append(alpha)
    chisq.append(chisqrfunc(d1h,d2h,data,C,M_min,M_1,alpha))


#param=[M_min_array,M_1_array,alpha_array]
neighbors=spatial.cKDTree(param)
#MIN=np.nanmin(chisq)
#idx=chisq.index(MIN)
#f=files[idx]
#print 'chisq:', MIN, f
#########################################
#name=f[8:]
#parameters=name.split("_")
#M_min=np.float(parameters[0])
#M_1=np.float(parameters[1])
#alpha=np.float(parameters[2])
x0=[13,13,1.2]
#points=[Mmin_array,M1_array,alpha_array]
#INT=interpolate.RegularGridInterpolator(points, chisq, method='linear', bounds_error=False, fill_value=None)

mass_file='/vol/fohlen11/fohlen11_1/bhernandez/chomp/mVector_PLANCK-SMT z: 3.0.txt'

mass_tmp=np.loadtxt(mass_file)
#mass=interpolate.InterpolatedUnivariateSpline(mass_tmp[:,0],mass_tmp[:,5])
mass=interpolate.InterpolatedUnivariateSpline(np.log(mass_tmp[:,0]),mass_tmp[:,6])



Nfile=open("N_24.2t24.4.dat", "a")
Mfile=open("Mhalo_24.2t24.4.dat", "a")
corrfile=open("correlations_24.2t24.4.dat", "a")

results=mcmc_fitting(x0, 500,neighbors)

##### Computing N and M_halo######

M_min=10**(results[0][0][0])
M_1=10**(results[0][1][0])
alpha=results[0][2][0]

M_min_errp=10**(results[0][0][0]+results[0][0][1])
M_1_errp=10**(results[0][1][0]+results[0][1][1])
alpha_errp=results[0][2][0]+results[0][2][1]

M_min_errm=10**(results[0][0][0]-results[0][0][2])
M_1_errm=10**(results[0][1][0]-results[0][1][2])
alpha_errm=results[0][2][0]-results[0][2][2]

N_tmp=np.loadtxt('N_24.2t24.4.dat')
M_tmp=np.loadtxt('Mhalo_24.2t24.4.dat')


size=len(N_tmp)
sel=N_tmp[size/3.:]
size_sel=len(sel)
sel.sort()
    
percentile=size_sel/4
N_errm=sel[size_sel/2]-sel[percentile]
N_errp=sel[3*percentile]-sel[size_sel/2]
N=np.median(sel)


size=len(M_tmp)
sel=M_tmp[size/3.:]
size_sel=len(sel)
sel.sort()
    
percentile=size_sel/4
M_halo_errm=sel[size_sel/2]-sel[percentile]
M_halo_errp=sel[3*percentile]
M_halo=np.median(sel)

#N_err=np.var(N_tmp[50:])
#M_halo_err=np.var(M_tmp[50:])

#N=inte.romberg(integrand_N_occnum,np.log(M_min),np.log(8.9*10**14),args=[M_1,alpha])/inte.romberg(integrand_Nden, np.log(M_min),np.log(8.9*10**14))

#M_halo=inte.romberg(integrand_M,np.log(M_min),np.log(8.9*10**14),args=[M_1,alpha])/inte.romberg(integrand_Mden, np.log(M_min),np.log(8.9*10**14),args=[M_1,alpha])


#N_errp=inte.romberg(integrand_N_occnum,np.log(M_min_errp),np.log(8.9*10**14),args=[M_1_errp,alpha_errp])/inte.romberg(integrand_Nden, np.log(M_min_errp),np.log(8.9*10**14))

#M_halo_errp=inte.romberg(integrand_M,np.log(M_min_errp),np.log(8.9*10**14),args=[M_1_errp,alpha_errp])/inte.romberg(integrand_Mden, np.log(M_min_errp),np.log(8.9*10**14),args=[M_1_errp,alpha_errp])

#N_errm=inte.romberg(integrand_N_occnum,np.log(M_min_errm),np.log(8.9*10**14),args=[M_1_errm,alpha_errm])/inte.romberg(integrand_Nden, np.log(M_min_errm),np.log(8.9*10**14))

#M_halo_errm=inte.romberg(integrand_M,np.log(M_min_errm),np.log(8.9*10**14),args=[M_1_errm,alpha_errm])/inte.romberg(integrand_Mden, np.log(M_min_errm),np.log(8.9*10**14),args=[M_1_errm,alpha_errm])
print "M_min: %s + %s - %s"%(results[0][0][0],results[0][0][1],results[0][0][2])
print "M_1: %s + %s - %s"%(results[0][1][0],results[0][1][1],results[0][1][2])
print "alpha: %s + %s - %s"%(results[0][2][0],results[0][2][1],results[0][2][2])
print "chi^2:", lnlike((M_min,M_1,alpha),neighbors)
print "N: %s + %s - %s"%(N,N_errp,N_errm)
print "M_halo: %s  + %s - %s"%(np.log10(M_halo),np.log10(M_halo_errp)-np.log10(M_halo),np.log10(M_halo)-np.log10(M_halo_errm))
print "IC:", IC
print "r_0:", r0
##############################
#sampler=results[1]
#plt.figure()
#samplesm= sampler.chain[:,:,0]
#samplesb = sampler.chain[:,:,1]
#samplesf = sampler.chain[:,:,2]
#plt.subplot(311)
#plt.plot(np.arange(len(samplesm)),samplesm)
#plt.ylabel('M_min')
#plt.subplot(312)
#plt.plot(np.arange(len(samplesb)),samplesb)
#plt.ylabel('M_1')
#plt.subplot(313)
#plt.plot(np.arange(len(samplesf)),samplesf)
#plt.ylabel('alpha')
##plt.savefig('check.png')
#plt.show()

#burnin = 50
#samples = sampler.chain[:, burnin:, :].reshape((-1, 3))
#fig = corner.corner(samples, labels=["$M_{min}$", "$M_1$", "$alpha$"])
##fig.savefig("line-triangle.png")
#fig.show()

####################################################
#(M_min,M_1,alpha)=parameters


data=np.loadtxt('/users/bhernandez/thesis/work/Wtheta_24.2t24.4_udropouts_weight_density')#'/vol/fohlen11/fohlen11_1/bhernandez/data/corr/udropouts/final/Wtheta_23t24_individual_weights')#/vol/fohlen11/fohlen11_1/bhernandez/data/corr/magnitude_bins/individual_udropouts/Wtheta_23.0t23.2_udropouts_weight')#/Wtheta_individualpointings_23t24')#'/vol/fohlen11/fohlen11_1/bhernandez/data/corr/udropouts/final/Wtheta_udropouts_m23t24_with_proper_weight')#'/vol/fohlen11/fohlen11_1/bhernandez/data/corr/magnitude_bins/udropouts_0.2/Wtheta_combined_24.2t24.4_udropouts') #'/vol/fohlen11/fohlen11_1/bhernandez/data/corr/udropouts/final/Wtheta_pointings_weights_small_scales_m23t24_RR.txt')#Wtheta_combined_small_scales_noregions_m23t24')
proc=[]
if not os.path.exists('corr_1h_%s_%s_%s'%(M_min,M_1,alpha)):
    arg1="./acf_1h_v2.exe -f corr_1h_%s_%s_%s -q QZ/z_PofZ_Udropouts_modified_v3.dat -mn %s -m1 %s -al %s"%(M_min,M_1,alpha,M_min,M_1,alpha)
    arg2="./acf_2h_v2.exe -f corr_2h_%s_%s_%s -q QZ/z_PofZ_Udropouts_modified_v3.dat -mn %s -m1 %s -al %s"%(M_min,M_1,alpha,M_min,M_1,alpha)
    proc.append(subprocess.Popen(arg1,shell='True'))
    proc.append(subprocess.Popen(arg2,shell='True'))
    proc[0].wait()
    proc[1].wait()
    
d1h=np.loadtxt('corr_1h_%s_%s_%s'%(M_min,M_1,alpha))
d2h=np.loadtxt('corr_2h_%s_%s_%s'%(M_min,M_1,alpha))
#proc=[]
#if not os.path.exists('corr_1h_%s_%s_%s'%(M_min_errp,M_1_errp,alpha_errp)):
    #arg1="./acf_1h_v2.exe -f corr_1h_%s_%s_%s -q QZ/z_PofZ_Udropouts_modified_v3.dat -mn %s -m1 %s -al %s"%(M_min_errp,M_1_errp,alpha_errp,M_min_errp,M_1_errp,alpha_errp)
    #arg2="./acf_2h_v2.exe -f corr_2h_%s_%s_%s -q QZ/z_PofZ_Udropouts_modified_v3.dat -mn %s -m1 %s -al %s"%(M_min_errp,M_1_errp,alpha_errp,M_min_errp,M_1_errp,alpha_errp)
    #proc.append(subprocess.Popen(arg1,shell='True'))
    #proc.append(subprocess.Popen(arg2,shell='True'))
    #proc[0].wait()
    #proc[1].wait()
    
#d1hp=np.loadtxt('corr_1h_%s_%s_%s'%(M_min_errp,M_1_errp,alpha_errp))
#d2hp=np.loadtxt('corr_2h_%s_%s_%s'%(M_min_errp,M_1_errp,alpha_errp))
#proc=[]
#if not os.path.exists('corr_1h_%s_%s_%s'%(M_min_errm,M_1_errm,alpha_errm)):
    #arg1="./acf_1h_v2.exe -f corr_1h_%s_%s_%s -q QZ/z_PofZ_Udropouts_modified_v3.dat -mn %s -m1 %s -al %s"%(M_min_errm,M_1_errm,alpha_errm,M_min_errm,M_1_errm,alpha_errm)
    #arg2="./acf_2h_v2.exe -f corr_2h_%s_%s_%s -q QZ/z_PofZ_Udropouts_modified_v3.dat -mn %s -m1 %s -al %s"%(M_min_errm,M_1_errm,alpha_errm,M_min_errm,M_1_errm,alpha_errm)
    #proc.append(subprocess.Popen(arg1,shell='True'))
    #proc.append(subprocess.Popen(arg2,shell='True'))
    #proc[0].wait()
    #proc[1].wait()
    
#d1hm=np.loadtxt('corr_1h_%s_%s_%s'%(M_min_errm,M_1_errm,alpha_errm))
#d2hm=np.loadtxt('corr_2h_%s_%s_%s'%(M_min_errm,M_1_errm,alpha_errm))




proc=[]
Wtheta_H=(d1h[:-1,1]+d2h[:-1,1])
theta=(d1h[:-1,0])
logtheta=np.log(data[1:14,0])#theta)
correlations = interpolate.InterpolatedUnivariateSpline(theta, Wtheta_H)
Wtheta=interpolate.InterpolatedUnivariateSpline(logtheta,correlations(data[1:14,0]))
av=[]
avp=[]
avm=[]
num=0
angle=[]
#idx=0
#for theta_bin in wtheta_ang.Bins():
    #logTMin=np.log(theta_bin.ThetaMin())
    #logTMax=np.log(theta_bin.ThetaMax())
    #TMin=theta_bin.ThetaMin()
    #TMax=theta_bin.ThetaMax()
    ##idx+=1
    ##print inte.romberg(function_corr,logTMin,logTMax,args=[Wtheta])/(TMax-TMin), data[num,1]
    #angle.append((TMin+TMax)/2.)
    #av.append(inte.romberg(function_corr,logTMin,logTMax,args=[Wtheta])/(TMax-TMin))

    #num+=1
    
#proc=[]
#Wtheta_H=(d1hp[:-1,1]+d2hp[:-1,1])
#theta=(d1hp[:-1,0])
#logtheta=np.log(data[1:14,0])#theta)
#correlations = interpolate.InterpolatedUnivariateSpline(theta, Wtheta_H)
#Wthetap=interpolate.InterpolatedUnivariateSpline(logtheta,correlations(data[1:14,0]))
##idx=0
#angle=[]
#num=0
#for theta_bin in wtheta_ang.Bins():
    #if 20>num>=1:
	#logTMin=np.log(theta_bin.ThetaMin())
	#logTMax=np.log(theta_bin.ThetaMax())
	#TMin=theta_bin.ThetaMin()
	#TMax=theta_bin.ThetaMax()
	##idx+=1
	##print inte.romberg(function_corr,logTMin,logTMax,args=[Wtheta])/(TMax-TMin), data[num,1]
	#angle.append((TMin+TMax)/2.)
	#avp.append(inte.romberg(function_corr,logTMin,logTMax,args=[Wthetap])/(TMax-TMin))

    #num+=1
    
#proc=[]
#Wtheta_H=(d1hm[:-1,1]+d2hm[:-1,1])
#theta=(d1hm[:-1,0])
#logtheta=np.log(data[1:14,0])#theta)
#correlations = interpolate.InterpolatedUnivariateSpline(theta, Wtheta_H)
#Wthetam=interpolate.InterpolatedUnivariateSpline(logtheta,correlations(data[1:14,0]))
##idx=0
#angle=[]
#num=0
#for theta_bin in wtheta_ang.Bins():
    #if 20>num>=1:
	#logTMin=np.log(theta_bin.ThetaMin())
	#logTMax=np.log(theta_bin.ThetaMax())
	#TMin=theta_bin.ThetaMin()
	#TMax=theta_bin.ThetaMax()
	##idx+=1
	##print inte.romberg(function_corr,logTMin,logTMax,args=[Wthetam])/(TMax-TMin), data[num,1]
	#angle.append((TMin+TMax)/2.)
	#avm.append(inte.romberg(function_corr,logTMin,logTMax,args=[Wthetam])/(TMax-TMin))

    #num+=1
    
    
    
    
av_tmp=np.loadtxt('correlations_24.2t24.4.dat')
size=len(av_tmp)
sel=av_tmp[size/3.:]
size_sel=len(sel)
for i in range(len(sel[0])):
    sel[:,i].sort()
    
percentile=size_sel*0.16#size_sel/4
av_errm=sel[percentile]
av_errp=sel[size_sel*(0.5+0.34)]
av=sel[size_sel/2]
#-sel[size_sel/2]

############10**
###########theta=(d1h[:,0])
#################lin=np.log10(av[4:8])
#################linang=np.log10(angle[4:8])
#################beta,A = curve_fit(func_fit, linang, lin)[0]
#################gamma=beta+1
#################r0_int=(10**A*(inte.romberg(pz_high,0, 5))**2/(inte.romberg(lambda z:pz_high(z)**2*((1+z)*cosmo_multi.angular_diameter_distance(z))**(1-gamma)*(c*cosmo_multi.E(z))**-1,0,5)*special.beta(1/2.,(gamma-1)/2.)))
#################r0=r0_int**(1/gamma)

#################sigma8g=72.*(r0/8.)**gamma/((3.-gamma)*(4.-gamma)*(6.-gamma)*2.**gamma)
#################sigma8z=cosmo_multi.sigma_m(8.0, redshift=3)**2#(cosmo_multi.luminosity_distance(3.)*0.9)**2 #
#################b2=sigma8g/sigma8z
#################IC=b2*IC_tmp
############correlations_test = interpolate.InterpolatedUnivariateSpline(theta, correlations)#interp1d(theta, correlations, kind='cubic')

###################RR=data[5:-7,-1]
###################IC=np.sum(RR*np.array(av))/np.sum(RR) #IC=(1+correlations)*np.sum(correlations)/(np.sum(RR)**(2))#


plt.errorbar(data[:-1,0]*60.,data[:-1,1],data[:-1,2]**(1/2.),fmt='x', markersize=10,lw=2)
plt.plot(d1h[:,0]*60.,d1h[:,1]+d2h[:,1] - IC,lw=2)
#plt.plot(data[1:20,0]*60.,av - IC,lw=2)
#plt.plot(d1h[:,0]*60., 0.12486203902693749**2*corr_low.correlation(d1h[:,0]*deg_to_rad)+0.87513796097306273**2*correlations-IC,'--')#1.97/20*      +18.03/20.*corr.correlation(data[:,0]/60.*deg_to_rad),'--')#delt[0]*data[:,0]+A[0],'--')
plt.fill_between(d1h[:,0]*60.,av_errp-IC,av_errm-IC, facecolor='0.75',alpha=0.5)
#plt.fill_between(data[1:20,0]*60.,av_errp-IC,av_errm-IC, facecolor='0.75',alpha=0.5)

plt.xscale('log')
plt.yscale('log')
plt.xlim(0.01,10)
plt.ylim(0.01,10)
plt.xlabel(r"$\theta$ [arcmin]",size=15)
plt.ylabel(r"w($\theta$)",size=15)
plt.title('24.2<r<24.4 udropouts')
#plt.text(5,10**1,r"$M_{min}$=%.3f$\pm$%.3f"%(np.log10(M_min),np.log10(errorpM_min)/np.log10(M_min))+ "\n" 
	 #+r"$M_1$=%.3f$\pm$%.3f"%(np.log10(M_1),np.log10(errorpM_1)/np.log10(M_1))+"\n"+r"$\alpha$=%.3f$\pm$%.3f"%(alpha,errorpalpha))
#plt.text(2.,10**0,r"$M_{min}$=%.3f+%.3f-%.3f"%(np.log10(M_min),np.log10(errorpM_min)/np.log10(M_min),np.log10(errormM_min)/np.log10(M_min))+ "\n" 
	 #+r"$M_1$=%.3f+%.3f-%.3f"%(np.log10(M_1),np.log10(errorpM_1)/np.log10(M_1),np.log10(errormM_1)/np.log10(M_1))+"\n"+r"$\alpha$=%.3f+%.3f-%.3f"%(alpha,errorpalpha,errormalpha))
plt.savefig('final_fit_242t244.eps', format='eps')
plt.show()


