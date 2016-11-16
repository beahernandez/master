#/usr/bin/python

from glob import glob
import subprocess

sample_list = ['z0.1t0.3', 'z0.3t0.5', 'z0.5t0.7', 'z0.7t0.9', 'z0.9t1.1',
               'z1.1t1.3', 'z0.1t1.3', 'udropouts', 'gdropouts']
# sample_list = ['udropouts']
n_dim_list = [32]

for n_dim in n_dim_list:
    for sample_name in sample_list:
        job = subprocess.Popen(
            'python /users/cmorrison/src/CFHTAnalysis/sys_som/som_map.py '
            '-ssom_K_var_ndim%i_iter100000_b0.006_l1.00.pkl '
            '-dsom_density_%s_var_ndim%i_iter100000_b0.006_l1.00_bootstrap.pkl --scale_variables '
            '--variance_scale'%
            (n_dim, sample_name, n_dim),
            shell = True)
        job.wait()
        
n_dim_list = [32]

for n_dim in n_dim_list:
    for sample_name in sample_list:
        job = subprocess.Popen(
            'python /users/cmorrison/src/CFHTAnalysis/sys_som/som_map.py '
            '-ssom_K_var_ndim%i_iter100000_b0.006_l1.00.pkl '
            '-dsom_density_%s_var_ndim%i_iter100000_b0.006_l1.00_bootstrap.pkl --scale_variables '
            '--variance_scale --use_errors'%
            (n_dim, sample_name, n_dim),
            shell = True)
        job.wait()