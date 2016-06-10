import time
from multiprocessing import Process, Queue, Array, Value, cpu_count
import sys
import numpy as np

class parallelization():
    """
    Parallelization Class for Function Calls.
    """
    def __init__(self, maximum_number_of_cores=cpu_count(), display=False):
        """
        Initialize a Parallelization Class.
        
        Parameters:
        ----------
        (optional) maximum_number_of_cores: (int) Maximal number of available CPUs, automatic assignment is
                                            all available ones.
        (optional) display:                 (bool) If set to TRUE, the calculated percentage of voxel will be 
                                            printed while the algorithm is running.
        """
        self.maximum_number_of_cores = maximum_number_of_cores
        self.display = display
        return None
        
    def worker(self, q, process_number, cpus, function, n, args, kwargs):
        done = False 
        while done != True:
            
            with self.calculations.get_lock():
                vox = sum(self.calculations)
                self.calculations[process_number] += 1
        
            # Everything done
            if vox >= n:
                done = True
                break
        
            # Calculate approximate time remaining for fitting
            percentage_realistic = int(vox*1.0/n*100.0)
            
            if percentage_realistic > self.percent.value:
                self.percent.value = percentage_realistic

                # Three-point estimation (PERT-Estimation)
                percentage_pessimistic = int(min(self.calculations)*self.number_of_cores*1.0/n*100.0)
                percentage_optimistic = int(max(self.calculations)*self.number_of_cores*1.0/n*100.0)
                
                if percentage_pessimistic == 0:
                    percentage_pessimistic = 1
                
                t_delta = time.time() - self.starting_time         
                time_remaining_optimistic = t_delta/percentage_optimistic*(100.0-percentage_optimistic)/60.0
                time_remaining_realistic = t_delta/percentage_realistic*(100.0-percentage_realistic)/60.0
                time_remaining_pessimistic = t_delta/percentage_pessimistic*(100.0-percentage_pessimistic)/60.0
                time_remaining = (time_remaining_optimistic + 4.0*time_remaining_realistic + time_remaining_pessimistic)/6.0 
                time_remaining = np.abs(time_remaining)
                if self.display == True:
                    time_remaining_seconds = int(time_remaining%1*60.0)
                    time_remaining_minutes = int(time_remaining)
                    sys.stdout.write('{:3.0f}%  {:4.0f}:{:02.0f} min remaining\n'.format(percentage_realistic, time_remaining_minutes, time_remaining_seconds))
    
            params = ()
            for param in args:
                if len(param) != 1:
                    params = params + (param[vox],)
                else:
                    params = params + (param[0],)
                    
            params_dict = {}
            for key, value in kwargs.iteritems():
                if len(value) != 1:
                    params_dict[key] = value[vox]
                else:
                    params_dict[key] = value[0]        
                    
                    
            if isinstance(function, list):
                result = function[vox](*params, **params_dict)
            else:
                result = function(*params, **params_dict)
    
            results = []
            results.append(vox)
            results.append(result)
            q.put(results)
        
    def start(self, function, n, *args, **kwargs):
        """"
        Starts the Parallelization of the Parallelization Class.
        
        Parameters:
        ----------
        function:                          (List) List of functions that should be executed, if always the same 
                                           function should be used, only the function name has to be assigned. 
                                           At iteration i function[i] of the list will be executed, together with the 
                                           belonging list element of args[i].
        n:                                 (int) Number of times the function should be called.
        (optional) *args:                  (Tuple of Lists) Parameter for executed function. If always the same parameter
                                           should be used, then a list containing one parameter [parameter] has to be
                                           assigned. Otherwise args[i] will be passed through to the function.
        (optional) **kwargs:               (Dictionary of Lists) Optional Keyword Parameter for the function. The Value 
                                           should always contain a list. If always the same parameter should be used,
                                           then a list containing one parameter kwarg=[parameter] has to be assigned. 
        (optional) multiple_return_values: (Boolean) **** ATTENTION: If definded, this variable needs to be the last one
                                           in the calling function, after args and kwargs! ****
                                           True, if number of returned variables of 'function()' is greater than one.
                                           If a list of functions  with a different amount of returned variables per function
                                           is called, multiple_return_values should be 'False'.
                                       
        Returns:
        --------
        List of return values from the excecuted function seperated by variables. The length of the variables equals n.
                            
        Example:
        -------
        models = p.start(TensorModel, 4, [gtab1, gtab2, gtab3, gtab4], multiple_return_values=True)
        fits, prediction = p.start([i.fit for i in models], 4, [data1, data2, data3, data4], [TE], sphere=[sphere])
        """
        
        multiple_return_values = False
        if 'multiple_return_values' in kwargs:
            multiple_return_values = kwargs['multiple_return_values']
            kwargs.pop('multiple_return_values')
        
        #Multicore Calculation
        self.number_of_cores = cpu_count()
        if self.maximum_number_of_cores < self.number_of_cores:
            self.number_of_cores = self.maximum_number_of_cores
        
        self.calculations = Array('i', self.number_of_cores)
        self.percent = Value('i', 0)

        self.percent.value = 0
        self.starting_time = time.time() 
        q = Queue()
        processes = []

        if self.display == True:
            print 'Parallelization starts on', self.number_of_cores, 'CPUs.'
        
        for i in range(self.number_of_cores):
            self.calculations[i] = 0
            processes.append(Process(target=self.worker, args=(q, i, self.number_of_cores, function, n, args, kwargs)))
            processes[-1].start()  
        
        return_values = [None] * n
        
        for i in range(n):
            results = q.get()
            vox = results[0]
            return_values[vox] = results[1]
        
        # Exit the completed processes
        for i in range(self.number_of_cores):
            processes[i].join()

        if self.display == True:
            sys.stdout.write('{:3.0f}%  {:4.0f}:{:02.0f} min remaining\n'.format(100.0, 0, 0))
            time_needed = (time.time() - self.starting_time)/60.0
            time_needed_seconds = int(time_needed%1*60.0)
            time_needed_minutes = int(time_needed)
            sys.stdout.write('\nTotal Time needed: {:5.0f}:{:02.0f} min\n'.format(time_needed_minutes, time_needed_seconds))
    
        # Transpose returned values 
        if multiple_return_values == True:
            return_values_seperated = np.asarray(return_values)
            return_values_tranformed = (return_values_seperated.T).tolist()
            return_values = tuple(return_values_tranformed)
            
        return return_values