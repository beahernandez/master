import numpy as np
import glob
import stomp
from astropy.io import fits
from matplotlib import pylab as plt

cat=glob.glob('/vol/fohlen11/fohlen11_1/bhernandez/data/cat/*zrgu_udropouts.cat')
data=[]
err=[]

mag_list=[23.1,23.3,23.5,23.7,23.9,24.1,24.3]
numgalaxies=[]
for mag in mag_list:
    number=[]
    for c in cat:
	print c

	cfht_cat = fits.getdata(c)
	if c[52]=='i':
	    mask = np.logical_and(
				cfht_cat['MAG_i'] >= mag-0.1,
				cfht_cat['MAG_i'] <  mag+0.1)
	else:
	    mask = np.logical_and(
				cfht_cat['MAG_y'] >= mag-0.1,
				cfht_cat['MAG_y'] <  mag+0.1)
	
	cfht_cat = cfht_cat[mask]
	
	m=stomp.Map('/vol/fohlen11/fohlen11_1/cmorrison/data/CFHTLenS/Maps/'+c[45:53]+'zrgu_finalmask_mosaic_stomp16384.map')
	A=m.Area()
	number.append(len(cfht_cat)/A)
	
    data=np.mean(number)
    err=np.var(number)
    numgalaxies.append([data,err])
    
plt.errorbar(mag_list,numgalaxies[:,0],numgalaxies[:,1],fmt='ko',lw=2,markersize=10)
plt.ylabel(r"N",size=15)
plt.xlabel(r"$r[mag AB]$",size=15)
plt.savefig('number_gal_dens.eps', format='eps')
plt.show()

    
#d=np.array(data)
#plt.plot(d[:,0],d[:,1])
#plt.errorbar(d[:,0],d[:,1],d[:,2], fmt='x')
#plt.show()
#stomp.Map.Write(map_tot,'combined_udropouts_weighted_CFHT_mag.map')