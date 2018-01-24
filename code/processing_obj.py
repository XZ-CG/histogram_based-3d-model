# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 11:35:39 2016

@author: lxz
"""
import numpy as np
import random

def loadOBJ(fliePath):
    numVertices = 0
    numUVs = 0
    numNormals = 0
#    numFaces = 0
    vertices = []
    uvs = []
    normals = []
    index = []
    vertexColors = []
#    faceVertIDs = []
#    uvIDs = []
#    normalIDs = []
    for line in open(fliePath, "r"):
        vals = line.split()
        if len(vals) == 0:
            continue
        if vals[0] == "v":
            v = map(float, vals[1:4])
            vertices.append(v)
            if len(vals) == 7:
                vc = map(float, vals[4:7])
                vertexColors.append(vc)
            numVertices += 1
        if vals[0] == "vt":#纹理顶点的坐标
            vt = map(float, vals[1:3])
            uvs.append(vt)
            numUVs += 1
        if vals[0] == "vn":
            vn = map(float, vals[1:4])
            normals.append(vn)
            numNormals += 1
        if vals[0] == "f":
            for f in vals[1:]:
                index.append(f)
                
            #下面程序是当输出obj文件中包括纹理坐标的时候
            #其格式为 f 123/12/134 1232/143/3342这种类型的时候
            #可以放心导入，如果仅仅包含法线以及顶点坐标请用上面的代码
#        if vals[0] == "f":
#            fvID = []
#            uvID = []
#            nvID = []
#            for f in vals[1:]:
#                w = f.split("/")
#                print w
#                if numVertices > 0:
#                    fvID.append(int(w[0])-1)
#                if numUVs > 0:
#                    uvID.append(int(w[1])-1)
#                if numNormals > 0:
#                    nvID.append(int(w[2])-1)
#            faceVertIDs.append(fvID)
#            uvIDs.append(uvID)
#            normalIDs.append(nvID)
#            numFaces += 1
    print "numVertices: ", numVertices
    print "numUVs: ", numUVs
    print "numNormals: ", numNormals
    return vertices,uvs,normals,vertexColors,index
#    print "numFaces: ", numFaces
#    return vertices, uvs, normals, faceVertIDs, uvIDs, normalIDs, vertexColors

def saveOBJ(filePath, vertices, uvs, normals, faceVertIDs, uvIDs, normalIDs, vertexColors):
    f_out = open(filePath, 'w')
    f_out.write("####\n")
    f_out.write("#\n")
    f_out.write("# Vertices: %s\n" %(len(vertices)))
    f_out.write("# Faces: %s\n" %(len( faceVertIDs)))
    f_out.write("#\n")
    f_out.write("####\n")
    for vi, v in enumerate( vertices ):
        vStr = "v %s %s %s"  %(v[0], v[1], v[2])
        if len( vertexColors) > 0:
            color = vertexColors[vi]
            vStr += " %s %s %s" %(color[0], color[1], color[2])
        vStr += "\n"
        f_out.write(vStr)
    f_out.write("# %s vertices\n\n"  %(len(vertices)))
    for uv in uvs:
        uvStr =  "vt %s %s\n"  %(uv[0], uv[1])
        f_out.write(uvStr)
    f_out.write("# %s uvs\n\n"  %(len(uvs)))
    for n in normals:
        nStr =  "vn %s %s %s\n"  %(n[0], n[1], n[2])
        f_out.write(nStr)
    f_out.write("# %s normals\n\n"  %(len(normals)))
    for fi, fvID in enumerate( faceVertIDs ):
        fStr = "f"
        for fvi, fvIDi in enumerate( fvID ):
            fStr += " %s" %( fvIDi + 1 )
            if len(uvIDs) > 0:
                fStr += "/%s" %( uvIDs[fi][fvi] + 1 )
            if len(normalIDs) > 0:
                fStr += "/%s" %( normalIDs[fi][fvi] + 1 )
        fStr += "\n"
        f_out.write(fStr)
    f_out.write("# %s faces\n\n"  %( len( faceVertIDs)) )
    f_out.write("# End of File\n")
    f_out.close()

def noise(P, sigma = 0.02):#给这个三维模型点添加正态的噪音
    xRange = np.max(P[:,0]) - np.min(P[:,0])
    yRange = np.max(P[:,1]) - np.min(P[:,1])
    zRange = np.max(P[:,2]) - np.min(P[:,2])
    noiseScale = 0.5 * np.linalg.norm( np.array([xRange, yRange, zRange]) )#求此向量的长度
    P_new = np.array( P )
    numVertices = len(P)
    for i in range(numVertices):
        P_new[i][0] += noiseScale * random.normalvariate(0,sigma)
        P_new[i][1] += noiseScale * random.normalvariate(0,sigma)
        P_new[i][2] += noiseScale * random.normalvariate(0,sigma)
    return P_new

#vertices,uvs,normals,vertexColors,index = loadOBJ('67P_C-G.obj')
#Point = np.array(vertices)
#P_n=noise(Point)
#np.savetxt("NoiseMesh.obj",P,delimiter=' ')
#P_noise = noise(P)
#saveOBJ('NoiseMesh.obj', P_noise, uvs, normals, faceVertIDs, uvIDs, normalIDs, vertexColors)