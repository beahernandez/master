#!/usr/bin/python    


import argparse
import h5py
import multiprocessing
from multiprocessing import Pool
from mvpa2.mappers.base import Mapper, accepts_dataset_as_samples
from mvpa2.suite import SimpleSOMMapper
from numba import jit
import numpy as np
import pickle
import stomp
from time import time

if __debug__:
    from mvpa2.base import debug
    
def _subsample_loop_wrapper(data):
    return _subsample_loop(data[0], data[1], data[2], data[3])

@jit(nopython = True)
def _subsample_loop(sub_sample, K, k, kshape):
        
    unit_delta = np.zeros(K.shape)
            
    # determine closest unit (as element coordinate)
    sq_diff = (K - sub_sample) ** 2
    min_dist = 1e32
    b = (-99, -99)
    for idx1 in xrange(kshape[0]):
        for idx2 in xrange(kshape[1]):
            tmp_sum = np.sum(sq_diff[idx1, idx2, :])
            if tmp_sum < min_dist:
                min_dist = tmp_sum
                b = (idx1, idx2)
                         
    # assumes 2D Kohonen layer
            
    # b = (np.divide(loc, kshape[1]).astype('int'), loc % kshape[1])
              
    # train all units at once by unfolding the kernel (from the
    # single quadrant that is precomputed), cutting it to the
    # right shape and simply multiply it to the difference of target
    # and all unit weights....
    for idx1 in xrange(kshape[0]):
        for idx2 in xrange(kshape[1]):
            #lower left
            if idx1 <= b[0] and idx2 <= b[1]:
                for idx3 in xrange(K.shape[2]):
                    unit_delta[idx1, idx2, idx3] += (
                        k[b[0] - idx1, b[1] - idx2] *
                        (sub_sample[idx3] - K[idx1, idx2, idx3]))
            #lower right
            elif idx1 > b[0] and idx2 <= b[1]:
                for idx3 in xrange(K.shape[2]):
                    unit_delta[idx1, idx2, idx3] += (
                        k[idx1 - b[0], b[1] - idx2] *
                        (sub_sample[idx3] - K[idx1, idx2, idx3]))
            #upper left
            elif idx1 <= b[0] and idx2 > b[1]:
                for idx3 in xrange(K.shape[2]):
                    unit_delta[idx1, idx2, idx3] += (
                        k[b[0] - idx1, idx2 - b[1]] *
                        (sub_sample[idx3] - K[idx1, idx2, idx3]))
            #upper right
            elif idx1 > b[0] and idx2 > b[1]:
                for idx3 in xrange(K.shape[2]):
                    unit_delta[idx1, idx2, idx3] += (
                        k[idx1 - b[0], idx2 - b[1]] *
                        (sub_sample[idx3] - K[idx1, idx2, idx3]))
                        
    return unit_delta
    
def _compute_density(som_idx_array, n_array, area_array,
                     field_array, som):
    
    density_values = np.zeros((som.K.shape[0], som.K.shape[1], 4))
    raw_value_list = []
    for idx1 in xrange(som.K.shape[0]):
        tmp_value_list = []
        for idx2 in xrange(som.K.shape[1]):
            tmp_value_list.append([])
        raw_value_list.append(tmp_value_list)
    
    for field_idx in xrange(171):
        
        # if field_idx % 10 == 0:
        #     print "Running Field", field_idx
        mask = (field_array == field_idx)
        
        tmp_area_array = area_array[mask]
        tmp_n_array = n_array[mask]
        mapped_array = som_idx_array[mask]
        field_mean_value = tmp_n_array.sum() / tmp_area_array.sum()
        
        for idx1 in xrange(som.K.shape[0]):
            for idx2 in xrange(som.K.shape[1]):
                
                mask = np.logical_and(idx1 == mapped_array[:,0],
                                      idx2 ==  mapped_array[:,1])
            
                if np.any(mask):
                    raw_value_list[idx1][idx2].append(
                        (tmp_n_array[mask].sum() / tmp_area_array[mask].sum()) /
                        field_mean_value - 1.0)
                    density_values[idx1, idx2, 2] += tmp_area_array[mask].sum()
    
    for idx1 in xrange(som.K.shape[0]):
        for idx2 in xrange(som.K.shape[1]):
        
            density_values[idx1, idx2, 0] = np.mean(raw_value_list[idx1][idx2])
            density_values[idx1, idx2, 1] = (np.std(raw_value_list[idx1][idx2]) / 
                                             np.sqrt(len(raw_value_list[idx1][idx2])))
            density_values[idx1, idx2, 3] = len(raw_value_list[idx1][idx2])
    
    # print density_values
    return density_values

def _compute_density_boot(som_idx_array, n_array, area_array,
                          field_array, som):
    
    density_values = np.zeros((som.K.shape[0], som.K.shape[1], 4))
    raw_value_list = []
    for idx1 in xrange(som.K.shape[0]):
        tmp_value_list = []
        for idx2 in xrange(som.K.shape[1]):
            tmp_value_list.append([])
        raw_value_list.append(tmp_value_list)
    
    for field_idx in xrange(171):
        
        # if field_idx % 10 == 0:
        #     print "Running Field", field_idx
        mask = (field_array == field_idx)
        
        tmp_area_array = area_array[mask]
        tmp_n_array = n_array[mask]
        mapped_array = som_idx_array[mask]
        field_n = tmp_n_array.sum()
        field_area = tmp_area_array.sum()
        
        for idx1 in xrange(som.K.shape[0]):
            for idx2 in xrange(som.K.shape[1]):
                
                mask = np.logical_and(idx1 == mapped_array[:,0],
                                      idx2 ==  mapped_array[:,1])
            
                if np.any(mask):
                    raw_value_list[idx1][idx2].append(
                        [tmp_n_array[mask].sum(), tmp_area_array[mask].sum(),
                         field_n, field_area])
                    density_values[idx1, idx2, 2] += tmp_area_array[mask].sum()
    
    for idx1 in xrange(som.K.shape[0]):
        for idx2 in xrange(som.K.shape[1]):
            
            value_len = len(raw_value_list[idx1][idx2])
            tmp_values = np.zeros(1000)
            for idx_value in xrange(1000):
                tmp_value_array = (np.array(raw_value_list[idx1][idx2])
                                   [np.random.randint(value_len, size =
                                                      value_len)])
                tmp_values[idx_value] = ((tmp_value_array[:,0].sum() /
                                         tmp_value_array[:,1].sum()) /
                                         (tmp_value_array[:,2].sum() /
                                         tmp_value_array[:,3].sum()) - 1.0)
            
            density_values[idx1, idx2, 0] = np.mean(tmp_values)
            density_values[idx1, idx2, 1] = np.std(tmp_values)
            density_values[idx1, idx2, 3] = len(raw_value_list[idx1][idx2])
    
    # print density_values
    return density_values


class SimpleSOMBootstrap(SimpleSOMMapper):
    
    def __init__(self, kshape, niter, learning_rate=0.005, batch_precent = 0.1,
                 iradius=None, distance_metric=None, initialization_func=None):
        
        self._b_prct = batch_precent
        
        SimpleSOMMapper.__init__(self, kshape, niter, learning_rate, iradius, 
                                 distance_metric, initialization_func)
        
    @accepts_dataset_as_samples
    def _train(self, samples):
        """Perform network training.

        Parameters
        ----------
        samples : array-like
            Used for unsupervised training of the SOM.
          
        Notes
        -----
        It is assumed that prior to calling this method the _pretrain method 
        was called with the same argument.  
        """

        # ensure that dqd was set properly
        dqd = self._dqd
        if dqd is None:
            raise ValueError("This should not happen - was _pretrain called?")
        
        sample_size = samples.shape[0]

        # units weight vector deltas for batch training
        # (height x width x #features)
        unit_deltas = np.zeros(self._K.shape, dtype='float')

        # for all iterations
        for it in xrange(1, self.niter + 1):
            # compute the neighborhood impact kernel for this iteration
            # has to be recomputed since kernel shrinks over time
            k = self._compute_influence_kernel(it, dqd)
            
            # sub_samples = np.random.shuffle(samples)[:int(sample_size *
            #                                               self._b_prct)]
            sub_samples = samples[:int(sample_size * self._b_prct)]

            # for all training vectors
            for s_idx, s in enumerate(sub_samples):
                # determine closest unit (as element coordinate)
                b = self._get_bmu(s)
                # train all units at once by unfolding the kernel (from the
                # single quadrant that is precomputed), cutting it to the
                # right shape and simply multiply it to the difference of target
                # and all unit weights....
                infl = np.vstack((
                        np.hstack((
                            # upper left
                            k[b[0]:0:-1, b[1]:0:-1],
                            # upper right
                            k[b[0]:0:-1, :self.kshape[1] - b[1]])),
                        np.hstack((
                            # lower left
                            k[:self.kshape[0] - b[0], b[1]:0:-1],
                            # lower right
                            k[:self.kshape[0] - b[0], :self.kshape[1] - b[1]]))
                               ))
                if s_idx == 0:
                    print b, infl.shape
                unit_deltas += infl[:, :, np.newaxis] * (s - self._K)

            # apply cumulative unit deltas
            self._K += unit_deltas

            if __debug__:
                debug("SOM", "Iteration %d/%d done: ||unit_deltas||=%g" %
                      (it, self.niter, np.sqrt(np.sum(unit_deltas ** 2))))

            # reset unit deltas
            unit_deltas.fill(0.)
            
    ##REF: Name was automagically refactored
    def _get_bmu(self, sample):
        """Returns the ID of the best matching unit.

        'best' is determined as minimal squared Euclidean distance between
        any units weight vector and some given target `sample`

        Parameters
        ----------
        sample : array
          Target sample.

        Returns
        -------
        tuple: (row, column)
        """
        # TODO expose distance function as parameter
        loc = np.argmin(((self.K - sample) ** 2).sum(axis=2))
        # assumes 2D Kohonen layer
        return (np.divide(loc, self.kshape[1]).astype('int'), loc % self.kshape[1])


class SimpleSOMSubSampleJIT(SimpleSOMMapper):
    
    def __init__(self, kshape, niter, learning_rate=0.005, batch_precent = 0.1,
                 iradius=None, distance_metric=None, initialization_func=None):
        
        self._b_prct = batch_precent
        
        SimpleSOMMapper.__init__(self, kshape, niter, learning_rate, iradius, 
                                 distance_metric, initialization_func)
        
    @accepts_dataset_as_samples
    def _train(self, samples):
        """Perform network training.
  
        Parameters
        ----------
        samples : array-like
            Used for unsupervised training of the SOM.
            
        Notes
        -----
        It is assumed that prior to calling this method the _pretrain method 
        was called with the same argument.  
        """
  
        # ensure that dqd was set properly
        dqd = self._dqd
        if dqd is None:
            raise ValueError("This should not happen - was _pretrain called?")
          
        sample_size = samples.shape[0]
  
        # units weight vector deltas for batch training
        # (height x width x #features)
  
        # for all iterations
        for it in xrange(1, self.niter + 1):
            # compute the neighborhood impact kernel for this iteration
            # has to be recomputed since kernel shrinks over time
            k = self._compute_influence_kernel(it, dqd)
              
            sub_samples = samples[
                np.random.randint(sample_size,
                                  size = int(sample_size * self._b_prct))]
  
            self._K += self._subsample_loop(sub_samples, self._K, k, self.kshape)
            
#     @accepts_dataset_as_samples
#     def _train(self, samples):
#         """Perform network training.
#  
#         Parameters
#         ----------
#         samples : array-like
#             Used for unsupervised training of the SOM.
#            
#         Notes
#         -----
#         It is assumed that prior to calling this method the _pretrain method 
#         was called with the same argument.  
#         """
#  
#         # ensure that dqd was set properly
#         dqd = self._dqd
#         if dqd is None:
#             raise ValueError("This should not happen - was _pretrain called?")
#  
#         # units weight vector deltas for batch training
#         # (height x width x #features)
#  
#         # for all iterations
#         tmp_K = self._iteration_loop(samples, self._b_prct, self._K, self.kshape,
#                                      dqd, self.radius, self.iter_scale,
#                                      self.lrate, self.niter)
#         self._K = tmp_K

    @staticmethod
    @jit(nopython = True)
    def _iteration_loop(samples, b_prct, K, kshape, dqd, radius, iter_scale,
                        lrate, niter):
        
        sample_size = samples.shape[0]
        
        for i_iter in xrange(1, niter + 1):
            
            ### Determine the current distance metric
            curr_max = radius * np.exp(-1.0 * i_iter / iter_scale)
            curr_lrate = lrate * np.exp(-1.0 * i_iter / iter_scale)
            ln_k = np.divide(dqd, 2.0 * curr_max * i_iter)
            k = curr_lrate * np.exp(-1.0 * ln_k)
            
            unit_deltas = np.zeros_like(K)
            
            ### begin sample loop for this iteration.
            for s_iter in xrange(int(sample_size * b_prct)):
                # Get the appropriate sample.
                s = samples[np.random.randint(sample_size)]
                
                # Check the current sample against the current nodes storing the
                # best match
                sq_diff = (K - s) ** 2
                min_dist = 1e32
                b = (-99, -99)
                for idx1 in xrange(kshape[0]):
                    for idx2 in xrange(kshape[1]):
                        tmp_sum = np.sum(sq_diff[idx1, idx2, :])
                        if tmp_sum < min_dist:
                            min_dist = tmp_sum
                            b = (idx1, idx2)
                
                # Update all the surrounding nodes using the sensitivity kernel
                
                for idx1 in xrange(kshape[0]):
                    for idx2 in xrange(kshape[1]):
                        #lower left
                        if idx1 <= b[0] and idx2 <= b[1]:
                            for idx3 in xrange(K.shape[2]):
                                unit_deltas[idx1, idx2, idx3] += (
                                    k[b[0] - idx1, b[1] - idx2] *
                                    (s[idx3] - K[idx1, idx2, idx3]))
                        #lower right
                        elif idx1 > b[0] and idx2 <= b[1]:
                            for idx3 in xrange(K.shape[2]):
                                unit_deltas[idx1, idx2, idx3] += (
                                    k[idx1 - b[0], b[1] - idx2] *
                                    (s[idx3] - K[idx1, idx2, idx3]))
                        #upper left
                        elif idx1 <= b[0] and idx2 > b[1]:
                            for idx3 in xrange(K.shape[2]):
                                unit_deltas[idx1, idx2, idx3] += (
                                    k[b[0] - idx1, idx2 - b[1]] *
                                    (s[idx3] - K[idx1, idx2, idx3]))
                        #upper right
                        elif idx1 > b[0] and idx2 > b[1]:
                            for idx3 in xrange(K.shape[2]):
                                unit_deltas[idx1, idx2, idx3] += (
                                    k[idx1 - b[0], idx2 - b[1]] *
                                    (s[idx3] - K[idx1, idx2, idx3]))
            
            K = np.add(K, unit_deltas)
            # for idx1 in xrange(K.shape[0]):
            #     for idx2 in xrange(K.shape[1]):
            #         for idx3 in xrange(K.shape[2]):
            #             K[idx1, idx2, idx3] +=  unit_deltas[idx1, idx2, idx3]
            
        return K
                            
    @staticmethod
    @jit(nopython = True)
    def _subsample_loop(sub_samples, K, k, kshape):
                
        unit_deltas = np.zeros(K.shape)
        
        # for all training vectors
        # for s_idx, s in enumerate(sub_samples):
        for s_idx in xrange(sub_samples.shape[0]):
            s = sub_samples[s_idx]
            
            # determine closest unit (as element coordinate)
            sq_diff = (K - s) ** 2
            min_dist = 1e32
            b = (-99, -99)
            for idx1 in xrange(kshape[0]):
                for idx2 in xrange(kshape[1]):
                    tmp_sum = np.sum(sq_diff[idx1, idx2, :])
                    if tmp_sum < min_dist:
                        min_dist = tmp_sum
                        b = (idx1, idx2)
                         
            # assumes 2D Kohonen layer
            
            # b = (np.divide(loc, kshape[1]).astype('int'), loc % kshape[1])
              
            # train all units at once by unfolding the kernel (from the
            # single quadrant that is precomputed), cutting it to the
            # right shape and simply multiply it to the difference of target
            # and all unit weights....
            for idx1 in xrange(kshape[0]):
                for idx2 in xrange(kshape[1]):
                    #lower left
                    if idx1 <= b[0] and idx2 <= b[1]:
                        for idx3 in xrange(K.shape[2]):
                            unit_deltas[idx1, idx2, idx3] += (
                                k[b[0] - idx1, b[1] - idx2] *
                                (s[idx3] - K[idx1, idx2, idx3]))
                    #lower right
                    elif idx1 > b[0] and idx2 <= b[1]:
                        for idx3 in xrange(K.shape[2]):
                            unit_deltas[idx1, idx2, idx3] += (
                                k[idx1 - b[0], b[1] - idx2] *
                                (s[idx3] - K[idx1, idx2, idx3]))
                    #upper left
                    elif idx1 <= b[0] and idx2 > b[1]:
                        for idx3 in xrange(K.shape[2]):
                            unit_deltas[idx1, idx2, idx3] += (
                                k[b[0] - idx1, idx2 - b[1]] *
                                (s[idx3] - K[idx1, idx2, idx3]))
                    #upper right
                    elif idx1 > b[0] and idx2 > b[1]:
                        for idx3 in xrange(K.shape[2]):
                            unit_deltas[idx1, idx2, idx3] += (
                                k[idx1 - b[0], idx2 - b[1]] *
                                (s[idx3] - K[idx1, idx2, idx3]))
                        
        return unit_deltas


class SimpleSOMSubSampleMultiProcessJIT(SimpleSOMMapper):
    
    def __init__(self, kshape, niter, learning_rate=0.005, batch_precent = 0.1,
                 n_processes = 4, iradius=None, distance_metric=None, initialization_func=None):
        
        self._b_prct = batch_precent
        self._n_process = n_processes
        
        SimpleSOMMapper.__init__(self, kshape, niter, learning_rate, iradius, 
                                 distance_metric, initialization_func)
        
    @accepts_dataset_as_samples
    def _train(self, samples):
        """Perform network training.
  
        Parameters
        ----------
        samples : array-like
            Used for unsupervised training of the SOM.
            
        Notes
        -----
        It is assumed that prior to calling this method the _pretrain method 
        was called with the same argument.  
        """
  
        # ensure that dqd was set properly
        dqd = self._dqd
        if dqd is None:
            raise ValueError("This should not happen - was _pretrain called?")
          
        sample_size = samples.shape[0]
        sub_sample_size = int(sample_size * self._b_prct)
  
        # units weight vector deltas for batch training
        # (height x width x #features)
        
        if self._n_process > multiprocessing.cpu_count():
            self._n_process = multiprocessing.cpu_count()
        
        n_submits = int(np.log(sample_size * self._b_prct) /
                        np.log(self._n_process))
        pool = Pool(self._n_process,
                    maxtasksperchild = sub_sample_size / self._n_process)
        print "n_submits =", n_submits
  
        # for all iterations
        for it in xrange(1, self.niter + 1):
            if it % (self.niter/10) == 0:
                print "Running iteration", it
            # compute the neighborhood impact kernel for this iteration
            # has to be recomputed since kernel shrinks over time
            k = self._compute_influence_kernel(it, dqd)
            
            subiter = SubSampleIterator(
                samples[np.random.randint(sample_size,
                        size = int(sample_size * self._b_prct))],
                self._K, k, self.kshape)
            
            
            unit_deltas = pool.imap(
                _subsample_loop_wrapper, subiter)
            
            # self._K += np.sum(unit_deltas, axis = 0)
            for unit_delta in unit_deltas:   
                self._K += unit_delta
            
#             output = []
#             for idx, sub_sample in enumerate(sub_samples):
#                 output.append(
#                     pool.map_unordered(_subsample_loop_wrapper,
#                                        args = (sub_sample, self._K, k,
#                                              self.kshape)))
#                 if len(output) >= n_submits or idx + 1 == sub_samples.shape[0]:
#                     for out in output:
#                         unit_deltas += out.get()
#                     del output
#                     output = []
            
            #self._K += self._subsample_loop(sub_samples, self._K, k, self.kshape)
        pool.close()
        pool.join()
    
class SubSampleIterator(object):
    
    def __init__(self, sub_samples, K, k, kshape):
        
        self._sub_samples = sub_samples
        self._K = K
        self._k = k
        self._kshape = kshape
        
        self._current_idx = 0
        self._end_idx = self._sub_samples.shape[0]
        
    def __iter__(self):
        return self
    
    def __next__(self):
        return self.next()
    
    def next(self):
        if self._current_idx < self._end_idx:
            self._current_idx += 1
            return [self._sub_samples[self._current_idx - 1],
                    self._K, self._k, self._kshape]
        else:
            raise StopIteration()
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sample_name', default = 'z0.1t0.3',
                        type = str, help = 'Sample to run.')
    parser.add_argument('--scale_variables', action = 'store_true',
                        help = 'Scale variables to range 0-1')
    parser.add_argument('--variance_scale', action = 'store_true',
                        help = 'Scale variables to variance of 1')
    parser.add_argument('-m', '--som_weights', default = '',
                        type = str, help = 'Weights from the SOM mapper')
    parser.add_argument('-b', '--n_bins', default = 10,
                        type = int, help = 'N iterations limit')
    parser.add_argument('-n', '--n_limit', default = 10,
                        type = int, help = 'N iterations limit')
    parser.add_argument('-i', '--input_hdf5_file', default='',
                        type = str, help='Input hdf5 file.')
    parser.add_argument('--batch_size', default=0.05,
                        type = float, help='Size of batch to learn on specified '
                        'in terms of percentage of the full sample.')
    parser.add_argument('--learning_rate', default=1.0,
                        type = float, help='Set the rate to learn.')
    parser.add_argument('--n_processes', default=4,
                        type = int, help='Number of child processes')
    parser.add_argument('--bootstrap', action = 'store_true')
    parser.add_argument('--name', default='',type = str,
                        help = '')
    args = parser.parse_args()
    
    sample_list = args.sample_name.split(',')
    print sample_list
    
    args = parser.parse_args()
    
    print "Loading data..."
    hdf5_file = h5py.File(args.input_hdf5_file)
    
    sys_all = np.empty((hdf5_file['sky/delta_depth'].shape[0], 11))
    
    sys_all[:,:5] = hdf5_file['sky/delta_depth'][...]
    sys_all[:,5:-1] = hdf5_file['seeing/delta_seeing'][...]
    sys_all[:,-1] = hdf5_file['extinction/delta_ext'][...]
    area_array = (hdf5_file['unmasked_area/unmasked'][...] *
                  stomp.Pixel.Area(4096))
    field_idx_array = hdf5_file['unmasked_area/field_id'][...]
    
    if args.scale_variables:
        print "Scaling Variables..."
        for idx in xrange(11):
            sys_min = np.min(sys_all[:,idx])
            sys_max = np.max(sys_all[:,idx])
            sys_all[:,idx] = (sys_all[:,idx] - sys_min)/(sys_max - sys_min)
    if args.variance_scale:
        for idx in xrange(11):
            sys_all[:,idx] /= np.std(sys_all[:,idx])
        print np.mean(sys_all, axis = 0)
            
    
    a_size = sys_all.shape[0]
    som = SimpleSOMSubSampleJIT((args.n_bins, args.n_bins), args.n_limit,
                                learning_rate = args.learning_rate /
                                (a_size * args.batch_size),
                                batch_precent = args.batch_size)
    print "Training SOM..."
    if args.som_weights == '':
        start_time = time()
        som.train(sys_all)
        print "JIT took", time() - start_time
        
        print som._K
        
        pkl_file = open('som_K_var_ndim%i_iter%i_b%.3f_l%.2f.pkl' %
                        (args.n_bins, args.n_limit, args.batch_size,
                         args.learning_rate),
                        'w')
        pickle.dump(som._K, pkl_file)
        pkl_file.close()
    else:
        pkl_file = open(args.som_weights)
        som._K = pickle.load(pkl_file)
        pkl_file.close()
        
        som.is_trained = True
    
    print "Retrieving SOM id's..."   
    # mask_ids = np.random.randint(sys_all.shape[0], size = sys_all.shape[0]/171) 
    map_array = som(sys_all)
    
    # weight_hdf5 = h5py.File('weight_som.hdf5')
    
    print "Computing Weights..."
    for sample_name in sample_list:
    
        print "\tfor %s..." % sample_name
        if not args.bootstrap:
            density_array = _compute_density(
                map_array, hdf5_file['%s/n_points' % sample_name],
                area_array, field_idx_array, som)
        else:
            density_array = _compute_density_boot(
                map_array, hdf5_file['%s/n_points' % sample_name],
                area_array, field_idx_array, som)
        if args.bootstrap:
            pkl_file = open('som_density_%s_var_ndim%i_iter%i_b%.3f_l%.2f_bootstrap_%s.pkl' %
                            (sample_name, args.n_bins, args.n_limit,
                             args.batch_size, args.learning_rate,args.name), 'w')
        else:
            pkl_file = open('som_density_%s_var_ndim%i_iter%i_b%.3f_l%.2f.pkl' %
                            (sample_name, args.n_bins, args.n_limit,
                             args.batch_size, args.learning_rate), 'w')
        pickle.dump(density_array, pkl_file)
        pkl_file.close()
        
#         try:
#             sample_grp = weight_hdf5.create_group(args.sample_name)
#             sample_grp.create_dataset('som_%s_ndim%i_iter%i' %
#                                       (sample_name, args.n_bins, args.n_limit),
#                                       data = weight_array)
#         except ValueError:
#             sample_grp = weight_hdf5[args.sample_name]
#             sample_grp.create_dataset('som_%s_ndim%i_iter%i' %
#                                       (sample_name, args.n_bins, args.n_limit),
#                                       data = weight_array)
    print "Writing weights."
    # weight_hdf5.close()
