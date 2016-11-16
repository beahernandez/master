#!/usr/bin/python

import argparse
import h5py
from mvpa2.mappers.base import Mapper, accepts_dataset_as_samples
from mvpa2.suite import SimpleSOMMapper
import numpy as np
import pickle
import stomp
from scipy.optimize import fsolve

field_filter_file = open(
    '/vol/fohlen11/fohlen11_1/cmorrison/data/CFHTLenS/chosen_dict.pkl')
field_filter_dict = pickle.load(field_filter_file)
field_filter_file.close()

def lognorm_root(x, mu, sigma):
        return np.array([np.exp(x[0] + x[1]**2 / 2) - mu,
                            np.exp(x[1]**2 + 2 * x[0]) *
                            (np.exp(x[1]**2) - 1) - sigma**2])

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--som_pkl', default = '',
                        type = str, help = 'Name of SOM pickle file.')
    parser.add_argument('-d', '--density_pkl', default = '',
                        type = str, help = 'Name of density pickle file')
    parser.add_argument('--scale_variables', action = 'store_true',
                        help = 'Scale variables to range of [0,1]')
    parser.add_argument('--variance_scale', action = 'store_true',
                        help = 'Scale variables to variance of 1')
    parser.add_argument('--path', default='',type = str,
                        help = '')
    parser.add_argument('--use_errors', action  = 'store_true',
                        help = 'Use errors to compute mean values. '
                        'Distribution is assumed to be lognormal '
                        'giving the measured mean and variance.')
    in_args = parser.parse_args()
    
    print "Loading data..."
    pkl_file = open(in_args.som_pkl)
    som_K_values = pickle.load(pkl_file)
    pkl_file.close()
    som = SimpleSOMMapper((som_K_values.shape[0], som_K_values.shape[1]),
                          1000)
    som._K = som_K_values
    som.is_trained = True
    
    pkl_file = open(in_args.density_pkl)
    den_values = pickle.load(pkl_file)
    pkl_file.close()
    
    if in_args.use_errors:
        M_array = np.empty_like(den_values[:,:,0])
        S_array = np.empty_like(den_values[:,:,0])
        for idx_1 in xrange(M_array.shape[0]):
            for idx_2 in xrange(M_array.shape[1]):
                ans = fsolve(lognorm_root,
                             x0 = [np.log(den_values[idx_1, idx_2, 0] + 1),
                                   den_values[idx_1, idx_2, 1]],
                             args = (den_values[idx_1, idx_2, 0] + 1,
                                     den_values[idx_1, idx_2, 1]),
                             full_output = True)
                if ans[2]:
                    M_array[idx_1, idx_2] = ans[0][0]
                    S_array[idx_1, idx_2] = ans[0][1]
                else:
                    print "Failed to find a lognormal solution."
                    raise ValueError
    
    cfht_data = h5py.File('/vol/fohlen11/fohlen11_1/bhernandez/data/map/weight_maps_som/magnitude_bins/CFHT_map_%s.hdf5'%in_args.path)#'
                          #'CFHT_map_23.0t23.2_udropouts.hdf5')
    sys_all = np.empty((cfht_data['sky/delta_depth'].shape[0], 11))
    
    sys_all[:,:5] = cfht_data['sky/delta_depth'][...]
    sys_all[:,5:-1] = cfht_data['seeing/delta_seeing'][...]
    sys_all[:,-1] = cfht_data['extinction/delta_ext'][...]
    area_array = (cfht_data['unmasked_area/unmasked'][...] *
                  stomp.Pixel.Area(4096))
    field_idx_array = cfht_data['unmasked_area/field_id'][...]
    
    if in_args.scale_variables:
        print "Scaling Variables..."
        for idx in xrange(11):
            sys_min = np.min(sys_all[:,idx])
            sys_max = np.max(sys_all[:,idx])
            sys_all[:,idx] = (sys_all[:,idx] - sys_min)/(sys_max - sys_min)
    if in_args.variance_scale:
        for idx in xrange(11):
            sys_all[:,idx] /= np.std(sys_all[:,idx])
    
    print 'Creating Maps...'
    for field_idx, field_name in enumerate(np.sort(field_filter_dict.keys())):
        
        map_file = open('/vol/fohlen11/fohlen11_1/cmorrison/data/CFHTLenS/'
                        'Extinction/%s_cfht_extinction_r4096.smap' % field_name)
        output_name = ('/vol/fohlen11/fohlen11_1/bhernandez/data/map/'
                       'weight_maps_som/magnitude_bins/%s/%s_som_%s' %
                       (in_args.path, field_name, in_args.density_pkl[12:-4]))
        if in_args.use_errors:
            output_name += '_err'
        output_name += '.map'
        output_map = open(output_name, 'w')
        n_line = 0
        
        tmp_som_idx_array = som(sys_all[field_idx_array == field_idx])
        if in_args.use_errors:
            tmp_weight_array = np.random.lognormal(
                M_array[tmp_som_idx_array[:,0], tmp_som_idx_array[:,1]],
                S_array[tmp_som_idx_array[:,0], tmp_som_idx_array[:,1]])
        else:
            tmp_weight_array = (den_values[tmp_som_idx_array[:,0],
                                           tmp_som_idx_array[:,1], 0]
                                + 1.0)
        
        for line in map_file:
            line_list = line.split()
            if float(line_list[4]) <= 0.0:
                continue
            
            output_map.writelines('%s %s %s %.8f\n' %
                                  (line_list[0], line_list[1], line_list[2],
                                   tmp_weight_array[n_line]))
            n_line += 1
        output_map.close()
