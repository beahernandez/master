import glob
import subprocess
#import os


#Searches for all catalog files
cat=glob.glob('/vol/fohlen11/fohlen11_1/hendrik/data/CFHTLenS/release_cats/W*zrgu_release_mask_plus.cat')

#Initiates the index to avoid too many processes at once
i=0
proc=[]

#Starts looping through the catalog files
for c in cat:
  #Defines the map and the output name for each catalog
  m='/vol/fohlen11/fohlen11_1/cmorrison/data/CFHTLenS/Maps/'+c[60:68]+'zrgu_finalmask_mosaic_stomp16384.map'
  out='/users/bhernandez/thesis/work/'+c[60:68]+'zrgu'#'/vol/fohlen11/fohlen11_1/bhernandez/data/test/'+c[60:68]+'zrgu'
  #Runs the script to obtain the autocorrelation
  if c[67]=='i':
    arg='python stomp_autocorrelation.py -c %s -m %s -o%s -n25 --z_min=0.1 --z_max=0.3 --mag_min=20 --mag_max=21' %(c,m,out)
  elif c[67]=='y':
    arg='python stomp_autocorrelation_y.py -c %s -m %s -o%s -n25 --z_min=0.1 --z_max=0.3 --mag_min=20 --mag_max=21' %(c,m,out)
  print i,arg
  proc.append(subprocess.Popen(arg,shell='True'))
  #os.system('python stomp_autocorrelation.py -c %s -m %s -o%s -n25 --z_min=0.1 --z_max=0.3 --mag_min=20 --mag_max=21' %(c,m,out))
  i=i+1
  
  #Checks the number of processes and if they are too many waits for them to finish
  if i>3:
    print 'Too many processes at once. Waiting for them to finish...'
    proc[0].wait()
    proc[1].wait()
    proc[2].wait()
    proc[3].wait()
    i=0
    proc=[]
    
  else:
    print 'Process number %i' %i 


  