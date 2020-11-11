import csv, time, random, math, sys
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy
import pycuda.driver as drv
 
def eucl_distance(point1, point2):
    if(len(point1) != len(point2)):
        raise Exception("Error: non comparable points")
 
    diff_sum = 0.0
    for i in range(len(point1)):
        diff = pow((float(point1[i]) - float(point2[i])), 2)
        diff_sum += diff
    final = math.sqrt(diff_sum)
    return final
 
 
def main(): 
    global cutoff, dim, dataset, num_of_clust, data,z
    z=0
    print ("Enter the number of clusters you want to make:")
    num_of_clust = int(input())
    with open('/content/modified_data_2020.csv', 'r') as f:
        reader = csv.reader(f)
        dataset = list(reader)
    initial = []
    dataset.pop(0)    
    data = dataset
    for i in range(num_of_clust):
        initial.append(dataset[i])
    f = open("cluster.txt", 'a')
    f.write(str(initial))
    initial = numpy.array(initial)
    initial = initial.astype(numpy.float32)
    num_points = []
    dim = []
 
    num_points.append(num_of_clust)
    num_points = numpy.array(num_points)
    num_points = num_points.astype(numpy.int32)
 
    dim.append(len(data[0]))
    dim = numpy.array(dim)
    dim = dim.astype(numpy.int32)
 
    cutoff = 0.2
    loop = 0
    clusters = []
    points = []
    for i in range(len(data)):
        points.append([0])
 
    data = numpy.array(data)
    data = data.astype(numpy.float32)
    points = numpy.array(points)
    points = points.astype(numpy.int32)
 
    points_gpu = cuda.mem_alloc(points.size * points.dtype.itemsize)
    cuda.memcpy_htod(points_gpu, points)
    data_gpu = cuda.mem_alloc(data.size * data.dtype.itemsize)
    cuda.memcpy_htod(data_gpu, data)
    initial_gpu = cuda.mem_alloc( initial.size * initial.dtype.itemsize)
    l = cuda.mem_alloc(dim.dtype.itemsize)
    cuda.memcpy_htod(l, dim)
    noc = cuda.mem_alloc(num_points.dtype.itemsize)
    cuda.memcpy_htod(noc, num_points) 
    compare_cutoff = True 
    str_time = time.time()
    while compare_cutoff:
        cuda.memcpy_htod(initial_gpu, initial)
        mod = SourceModule("""  
            __global__ void kmeans(float * a, float *c, int * d, int * len, int * noc)
            {
                int idx = blockIdx.x;
            int li;
            float least = 99999999;
            for(int i = 0; i< noc[0]; i++){
                float sum = 0.0;      
                for(int j = 0; j< len[0]; j++){
                    sum += ((a[i*len[0] + j] - c[idx*len[0] +j])* (a[i*len[0] + j] - c[idx*len[0] +j]));
                }
        
                if( sum < least){
                    least  = sum;
                    li = i;   
                }
            } 
            d[idx] = li;      
            }
            """)
        func = mod.get_function("kmeans") 
        func(initial_gpu, data_gpu, points_gpu, l, noc, block=(1, 1, 1), grid=(len(data), 1, 1))  
        pc = numpy.empty_like(points)
        cuda.memcpy_dtoh(pc, points_gpu)
        nm = numpy.empty_like(num_points)
        cuda.memcpy_dtoh(nm,noc)
        no = []
        total = []
        for i in range(len(initial)):
            no.append(0)
            total.append([])
            for j in range(len(initial[0])):
                total[-1].append(0)
        z+=1
        
        for i in range(len(pc)):
            no[int(pc[i][0])%4] += 1
            for j in range(len(initial[0])):
                total[int(pc[i][0]) % 4][j] += float(data[i][j])
        
        for i in range(len(total)):
            if(no[i] != 0):
                for j in range(len(initial[0])):
                    total[i][j] /= no[i]
        flag = 0    
        for i in range(len(total)):
            if eucl_distance(total[i], initial[i]) > cutoff:
                flag += 1
        if flag == 0:
            compare_cutoff = False
            print (total)
            f.write(str(total))
        else:
            total = numpy.array(total)
            initial = total.astype(numpy.float32)
    print ("Execution time %s seconds" % (time.time() - str_time))
    f.close()
if __name__ == "__main__":
    str_time = time.time()
    main()

