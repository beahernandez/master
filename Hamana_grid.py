import numpy as np
#import os
#from matplotlib import pylab as plt
#from scipy.interpolate import interp1d
import subprocess
import os.path

a=np.linspace(1.*10**(10.),5.*10**(13.),10)
i=0
proc=[]
for M_min in np.concatenate([np.logspace(10.,12.5,10),a[1:]]):
    for M_1 in np.concatenate([np.logspace(10.,12.5,10),a[1:]]):
    #np.logspace(10.,12.5,10):
	for alpha in np.linspace(1.0,2.0,10):
	    if not os.path.exists('corr_1h_%s_%s_%s'%(M_min,M_1,alpha)):
		if i<4:
		
	#for M_1 in np.linspace(3.1*10**(12.),4.5*10**(12.),5):
	#for M_min in np.linspace(5.1*10**(11.),8.*10**(11.),5):
	    #for alpha in np.arange(1.,1.1,0.05):
		    
		    arg1="nice -n15 ../acf_1h_v2.exe -f corr_1h_%s_%s_%s -q ../QZ/z_PofZ_Udropouts_modified_v3.dat -mn %s -m1 %s -al %s"%(M_min,M_1,alpha,M_min,M_1,alpha)
		    arg2="nice -n15 ../acf_2h_v2.exe -f corr_2h_%s_%s_%s -q ../QZ/z_PofZ_Udropouts_modified_v3.dat -mn %s -m1 %s -al %s"%(M_min,M_1,alpha,M_min,M_1,alpha)
		    proc.append(subprocess.Popen(arg1,shell='True'))
		    proc.append(subprocess.Popen(arg2,shell='True'))
		    i+=2
		    
		else:
		    proc[0].wait()
		    proc[1].wait()
		    proc[2].wait()
		    proc[3].wait()
		    #proc[4].wait()
		    #proc[5].wait()
		    proc=[]

		    arg1="nice -n15 ../acf_1h_v2.exe -f corr_1h_%s_%s_%s -q ../QZ/z_PofZ_Udropouts_modified_v3.dat -mn %s -m1 %s -al %s"%(M_min,M_1,alpha,M_min,M_1,alpha)
		    arg2="nice -n15 ../acf_2h_v2.exe -f corr_2h_%s_%s_%s -q ../QZ/z_PofZ_Udropouts_modified_v3.dat -mn %s -m1 %s -al %s"%(M_min,M_1,alpha,M_min,M_1,alpha)
		    proc.append(subprocess.Popen(arg1,shell='True'))
		    proc.append(subprocess.Popen(arg2,shell='True'))
		    i=2
	    else:
		print 'Done corr_1h_%s_%s_%s'%(M_min,M_1,alpha)