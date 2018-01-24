import math
import os
from datetime import datetime
import time

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from pycuda import compiler

feature_gpu_template = """
#include <math.h>
#define PI 3.14159
struct cuVector {
    float x;
    float y;
    float z;
    __device__ cuVector(float a, float b, float c) : x(a), y(b), z(c){}
    __device__ float operator*(cuVector& a) {
        return x*a.x + y*a.y + z*a.z;
    }
    __device__ cuVector operator/(cuVector& a){
        return cuVector(x / a.x, y / a.y, z / a.z);
    }
    __device__ cuVector operator^(cuVector& a){
        return cuVector(y * a.z - z * a.y, z * a.x - x * a.z, x * a.y - y * a.x);
    }
    __device__ cuVector operator-(cuVector& a){
        return cuVector(x * a.x, y * a.y, z * a.z);
    }
};

__device__ float compute_feature(cuVector n1,cuVector n2,cuVector dis,float internal,float e1,float e2,float e3,float e4)
{
    float w1 = fabs(n1 * dis);
    float w2 = fabs(n2 * dis);
    float d = sqrt(dis * dis);
    float a = 0.0;
    float e = (e1+e2)/2;
    float f = (e3+e4)/2;
    int a0,a1,a2,a3,a4,a5;
    if(d < 1e-3){
        return 0;
    }
    cuVector tmp(dis.x / d,dis.y / d, dis.z / d);
    unsigned int final;
    if(w1 <= w2){
        cuVector v1 = dis ^ n1;
        float v2 = sqrt(v1 * v1);
        if(v2 != v2){
            return 0;
        }
        cuVector v(v1.x / v2, v1.y / v2, v1.z / v2);
        cuVector w = n1 ^ v;
        float x = w * n2;
        float y = n1 * n2;
        if(x != x || y != y){
            return 0;
        }
        if(x > 0 && y > 0){
            a = atan(y / x);}
        else if(x < 0){
            a = atan(y / x) + PI;}
        else if(x > 0 && y < 0){
            a = atan(y / x) + PI *2;}
        float b = v * n2;
        float c = n1 * tmp;
        if(b != b || c != c){
            return 0;
        }
        a0 = ceil(5 * a / (2 * PI));
        a1 = ceil((b + 1.1) / 0.44);
        a2 = ceil((c + 1.1) / 0.44);
        a3 = ceil(d / internal);
        a4 = ceil(e);
        a5 = ceil(f);
        if(1<=a0<=5 &&1<=a1<=5 &&1<=a2<=5 &&1<=a3<=5 &&1<=a4<=5 &&1<=a5<=5){
            final= a0 * 3125 +  a1 * 625 +  a2 * 125 + a3 * 25 + a4 * 5+ a5 - 3905;
            }
        else{
            final = 0;
        }
    }
    else{
        cuVector v1 = dis ^ n2;
        float v2 = sqrt(v1 * v1);
        if(v2 != v2){
            return 0;
        }
        cuVector v(v1.x / v2, v1.y / v2, v1.z / v2);
        cuVector w = n2 ^ v;
        float x = w * n1;
        float y = n2 * n1;
        if(x != x || y != y){
            return 0;
        }
        if(x > 0 && y > 0){
            a = atan(y / x);}
        else if(x < 0){
            a = atan(y / x) + PI;}
        else if(x > 0 && y < 0){
            a = atan(y / x) + PI *2;}
        float b = v * n1;
        float c = n2 * tmp;
        if(b!=b || c!=c){
            return 0;
        }
        a0 = ceil(5 * a / (2 * PI));
        a1 = ceil((b + 1.1) / 0.44);
        a2 = ceil((c + 1.1) / 0.44);
        a3 = ceil(d / internal);
        a4 = ceil(e);
        a5 = ceil(f);
        if(1<=a0<=5 && 1<=a1<=5 && 1<=a2<=5 && 1<=a3<=5 && 1<=a4<=5 && 1<=a5<=5){
            final= a0 * 3125 +  a1 * 625 +  a2 * 125 + a3 * 25 + a4 * 5+ a5 - 3905;
            }
        else{
            final = 0;
        }
    }
    return final;
}

__global__ void doublify(float *data,int *light,int N,float internal,int length,int row)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int id = row * N + idy;
    if(idx < length && idy < row){
        cuVector n1(data[idx * 8 + 3], data[idx * 8 + 4], data[idx * 8 + 5]);
        cuVector n2(data[id * 8 + 3], data[id * 8 + 4], data[id * 8 + 5]);
        float a = data[idx * 8] - data[id * 8];
        float b = data[idx * 8 + 1] - data[id * 8 + 1];
        float c = data[idx * 8 + 2] - data[id * 8 + 2];
        cuVector dis(a,b,c);
        float e1 = data[idx * 8 + 6];
        float e2 = data[id * 8 + 6];
        float e3 = data[idx * 8 + 7];
        float e4 = data[id * 8 + 7];
        unsigned int bin_num = compute_feature(n1,n2,dis,internal,e1,e2,e3,e4);
        light[idy*length + idx] = bin_num;
        }
}
    """

def histogram(light):
    grid_gpu_template = """
    __global__ void grid(int *values, int size, int *temp_grid)
    {
         unsigned int id = threadIdx.x;
         int i,bin;
         for(i=id;i<size;i+=blockDim.x){
             bin=values[i];
             if (values[i]==%(interv)s){
                values[i]=%(interv)s-1;
             }
             temp_grid[id*%(interv)s+bin]+=1.0;
         }
    }
    """

    reduction_gpu_template = """
    __global__ void reduction(int *temp_grid, int *his)
    {
         unsigned int id = blockIdx.x*blockDim.x+threadIdx.x;
         if(id<%(interv)s){
             for(int i=0;i<%(max_number_of_threads)s;i++){
                 his[id]+=temp_grid[id+%(interv)s*i];
             }
         }
    }
    """
    number_of_points = len(light)
    max_number_of_threads = 1024
    interv = 15626

    blocks = interv / max_number_of_threads
    if interv % max_number_of_threads != 0:
        blocks += 1

    grid_gpu = grid_gpu_template % {
        'interv': interv,
    }
    mod_grid = compiler.SourceModule(grid_gpu)
    grid = mod_grid.get_function("grid")

    reduction_gpu = reduction_gpu_template % {
        'interv': interv,
        'max_number_of_threads': max_number_of_threads,
    }
    mod_redt = compiler.SourceModule(reduction_gpu)
    redt = mod_redt.get_function("reduction")
    values_gpu = gpuarray.to_gpu(light)
    temp_grid_gpu = gpuarray.zeros((max_number_of_threads, interv), dtype=np.int32)
    hist = np.zeros(interv, dtype=np.int32)
    hist_gpu = gpuarray.to_gpu(hist)
    grid(values_gpu, np.int32(number_of_points), temp_grid_gpu, grid=(1, 1), block=(max_number_of_threads, 1, 1))
    redt(temp_grid_gpu, hist_gpu, grid=(blocks, 1), block=(max_number_of_threads, 1, 1))
    hist = hist_gpu.get()
    return hist

def interval_distance(name):
    data = np.loadtxt(name)
    length = data.shape[0]
    long_pair = [math.sqrt(np.dot(data[i][:3],data[i][:3])) for i in xrange(length)]
    long_pair_index = long_pair.index(max(long_pair))
    distance = 0.0
    for i in xrange(length):
        sur_pair = data[long_pair_index][:3]-data[i][:3]
        sur_pair_dis = math.sqrt(np.dot(sur_pair,sur_pair))
        if sur_pair_dis >= distance:
            distance = sur_pair_dis
    return distance

def compute(data, N, internal, row, col, length):
    print(N)
    stat = time.clock() * 1e3
    d_data = gpuarray.to_gpu(data)
    d_light = gpuarray.zeros((row*length), np.int32)
    mod_doublify = compiler.SourceModule(feature_gpu_template)
    func = mod_doublify.get_function("doublify")
    func(d_data,d_light,np.int32(N),np.float32(internal),np.int32(length),np.int32(row),block=(1024,1,1),grid=(col,row))
    light = d_light.get()
    hist = histogram(light)
    print('Time used to grid with CPU:', time.clock() * 1e3 - stat, ' ms')
    return hist
    # his_CPU = np.histogram(light,bins=his_inter)[0]
    # return his_CPU
def third_step(test_name,load_model = 'Yes'):
    '''
    @count means the input file for classify
    @load_model means whether to use the old training dataset of model
    '''
    if load_model == 'Yes':
        model = pickle.load(open("temp.txt",'r'))
    else:
        input_list = Create_input_data() 
        input_label = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']
        model = sk.RandomForestClassifier(n_estimators = 1000)
        model.fit(input_list,input_label)
        pickle.dump(model,open("temp.txt",'w'))
    
    x_test = Create_seq_data(test_name)
    predicted = model.predict(x_test)
    return predicted[0]
	
def main():
    start = datetime.now()
    os.chdir("E:\\pycharm\\fast_histogram\\code")
    name = ["test_1.txt"]

    for name_asteroid in name:
        print(name_asteroid)
        data = np.loadtxt(name_asteroid, np.float32)
        internal = interval_distance(name_asteroid)
        length = data.shape[0]
        col = length / 1024 + 1
        limit = 100000 / col
        row = length if length <= limit else int(math.ceil(length / (length / limit + 1)))
        N = length / limit + 1
        print(length, row, col * 1024, N)
        hist = np.zeros((15626L,))
        for i in xrange(0, N):
            hist += compute(data, i, internal, row, col, length)
        np.savetxt('f%s' % (name_asteroid), hist, fmt="%d")
		result = third_step('f%d_0.txt'%count,load_model = 'Yes')
    foo.write("".join([name_asteroid,' ',result,' ',str(length),'\n'])) 
    print(datetime.now() - start)

if __name__ == "__main__":
    main()