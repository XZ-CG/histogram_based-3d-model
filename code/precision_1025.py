# -*- coding: utf-8 -*-
"""
Created on Tue May 10 16:45:52 2016

@author: lxz
"""
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import os
import math
import numpy as np
from collections import Counter
from datetime import datetime

mod = SourceModule("""
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

__device__ float compute_feature(cuVector n1,cuVector n2,cuVector dis,float internal,float e1,float e2)
{
    float w1 = fabs(n1 * dis);
    float w2 = fabs(n2 * dis);
    float d = sqrt(dis * dis);
    float a = 0.0;
    float e = (e1+e2)/2;
    if(d < 1e-3){
        return 5000;
    }
    cuVector tmp(dis.x / d,dis.y / d, dis.z / d);
    unsigned int final;
    if(w1 <= w2){
        cuVector v1 = dis ^ n1;
        float v2 = sqrt(v1 * v1);
        if(v2 != v2){
            return 4000;
        }
        cuVector v(v1.x / v2, v1.y / v2, v1.z / v2);
        cuVector w = n1 ^ v;
        float x = w * n2;
        float y = n1 * n2;
        if(x != x || y != y){
            return 2000;
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
            return 3000;
        } 
        final=ceil(5 * a / (2 * PI)) * 10000 + ceil((b + 1.1) / 0.44) * 1000 + ceil((c + 1.1) / 0.44) * 100 + ceil(d / internal)*10 + ceil(e);
 //       return d;
    }
    else{
        cuVector v1 = dis ^ n2;
        float v2 = sqrt(v1 * v1);
        if(v2 != v2){
            return 4000;
        }
        cuVector v(v1.x / v2, v1.y / v2, v1.z / v2);
        cuVector w = n2 ^ v;
        float x = w * n1;
        float y = n2 * n1;
        if(x != x || y != y){
            return 2000;
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
            return 3000;
        } 
        final=ceil(5 * a / (2 * PI)) * 10000 + ceil((b + 1.1) / 0.44) * 1000 + ceil((c + 1.1) / 0.44) * 100 + ceil(d / internal)*10 + ceil(e);
 //       return 7000;
    }
    return final;
}

__global__ void doublify(float *data,float *light,int N,float internal,int length,int row)
{ 
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int id = row * N + idy;
    if(idx < length && idy < length && idy < row){
        cuVector n1(data[idx * 7 + 3], data[idx * 7 + 4], data[idx * 7 + 5]);
        cuVector n2(data[id * 7 + 3], data[id * 7 + 4], data[id * 7 + 5]);
        float a = data[idx * 7] - data[id * 7];
        float b = data[idx * 7 + 1] - data[id * 7 + 1];
        float c = data[idx * 7 + 2] - data[id * 7 + 2];
        cuVector dis(a,b,c); 
        float e1 = data[idx * 7 + 6];
        float e2 = data[id * 7 + 6];
        unsigned int bin_num = compute_feature(n1,n2,dis,internal,e1,e2);
        light[idy*length + idx] = bin_num;
        }
}
  """)

def speedtest(func, *args, **kw):
    start = cuda.Event()
    end = cuda.Event()
    start.record()
    func(*args, **kw)
    end.record()
    end.synchronize()
    secs = start.time_till(end)
    print("GPU time: %fms" % secs)

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

def compute(count,data,N,internal,row,col,length): 
    print N
    d_data = cuda.mem_alloc(data.nbytes)
    cuda.memcpy_htod(d_data, data)
    
    light = np.zeros((row, length), np.float32)
    d_light = cuda.mem_alloc(light.nbytes)
    
    func = mod.get_function("doublify")
    speedtest(func, d_data, d_light,np.int32(N),np.float32(internal),np.int32(length),np.int32(row), block=(1024,1,1), grid=(col,row))
    cuda.memcpy_dtoh(light, d_light)
    d_light.free()
    d_data.free() 
    fo=open('f%d_%d.txt'%(count,N),"w")
    for i in xrange(row):
        dic=Counter(light[i])
        for k,v in dic.items():
            fo.write("".join([str(k),':',str(v),' ']))
        fo.write('\n')
    fo.close()

def main():
    os.chdir("C:\\Users\\lxz\\Desktop\\tmp2")
    start = datetime.now()
    name={'Ida.txt':8,'Ceres.txt':1,'Chury.txt':2,'Gaspra.txt':5,'Geogra.txt':6,'Hartley.txt':7,'Steins.txt':15,'Tempel.txt':16,'Toutatis.txt':18,'Wild.txt':20}
    for name_asteroid,count in name.iteritems():
        data = np.loadtxt(name_asteroid,np.float32)
        internal = interval_distance(name_asteroid)
        length = data.shape[0]
        col = length / 1024 + 1
        limit = 150000 / col
        row = length if length <= limit else limit
        N = length / row + 1
        print length,row,col*1024,N
        for i in xrange(0,N):
            compute(count,data,i,internal,row,col,length)   
    print datetime.now()-start
        
#    name = 'Gaspra Thomas_meshlab_curvature.txt'       
#    data = np.loadtxt(name,np.float32)
#    internal = interval_distance(name)
#    length = data.shape[0]
#    col = length / 1024 + 1
#    limit = 150000 / col
#    row = length if length <= limit else limit
#    N = length / row + 1
#    count = 100
#    print length,row,col*1024,N
#    for i in xrange(0,N):
#        compute(count,data,i,internal,row,col,length)    
#    print datetime.now()-start

if __name__=="__main__":
    main()  



