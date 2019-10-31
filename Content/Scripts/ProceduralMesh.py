from unreal_engine import  FVector, FRotator, FColor
from unreal_engine.classes import ProceduralMeshComponent

import numpy as np
import ObjLoader
from importlib import reload

reload(ObjLoader)

class CustomProceduralMeshComponent(ProceduralMeshComponent):
    """A ProceduralMeshComponent with the ability to convert RenderableArray geometry"""
    def __init__(self):
        self.CurrentMeshSectionIndex = 0

    def ReceiveBeginPlay(self):
        """Called when the actor is beginning play, or the world is beginning play"""
        self.CurrentMeshSectionIndex = 0

    # For test
    def updateself(self):
        uvArray = self.uvArray
        colorArray = self.colorArray
        normalArray = self.normalArray
        vertexArray = self.vertexArray

        for i in range(len(vertexArray)):
            vertexArray[i] += FVector(0,0.00001,0)

        self.UpdateMeshSection(self.CurrentMeshSectionIndex, vertexArray, normalArray, UV0=uvArray, VertexColors=colorArray)

    def update_normal(self, update_vertices):
        vertices = np.array(update_vertices)
        faces =  np.array(self.faces)

        #Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
        norm = np.zeros( vertices.shape, dtype=vertices.dtype )
        #Create an indexed view into the vertex array using the array of three indices for triangles
        tris = vertices[faces]
        #Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle             
        n = np.cross( tris[::,1 ] - tris[::,0]  , tris[::,2 ] - tris[::,0] )
        # n is now an array of normals per triangle. The length of each normal is dependent the vertices, 
        # we need to normalize these, so that our next step weights each normal equally.
        self.normalize_v3(n)
        # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
        # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle, 
        # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
        # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
        norm[ faces[:,0] ] += n
        norm[ faces[:,1] ] += n
        norm[ faces[:,2] ] += n
        self.normalize_v3(norm)
        return norm.tolist()

    def normalize_v3(self, arr):
        ''' Normalize a np array of 3 component vectors shape=(n,3) '''
        lens = np.sqrt( arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2 )
        arr[:,0] /= lens
        arr[:,1] /= lens
        arr[:,2] /= lens                
        return arr

    def update(self, vertices):
        if len(self.vertexArray) != len(vertices) and  len(self.vertexArray)*3 != len(vertices):
            print("***** update vertices number not equal to original************")
            return

        uvArray = self.uvArray
        colorArray = self.colorArray


        # optimize this later
        vertices2D = [] 
        vertexArray = []
        for i in range(0, len(vertices), 3):
            vertexArray.append(FVector(vertices[i], vertices[i+1], vertices[i+2]))
            vertices2D.append([vertices[i], vertices[i+1], vertices[i+2]])

        # update normla with naive method, cost lots of time
        normalArray = self.update_normal(vertices2D)
        normalArrayForRender = []

        for i in range(0, len(normalArray)):
            normalArrayForRender.append(FVector(normalArray[i][0], normalArray[i][1], normalArray[i][2]))


        self.UpdateMeshSection(self.CurrentMeshSectionIndex, vertexArray, normalArrayForRender, UV0=uvArray, VertexColors=colorArray)

    def import_renderable(self, filename):
        """Adds the specified renderable as a mesh section to this procedural mesh component"""
        # obj = ObjLoader.OBJ(filename)
        obj = ObjLoader.OBJ(filename)

        self.vertexArray = obj.vertices        
        self.indexArray = obj.faces
        self.faces = []
        for i in range(0, len(obj.faces), 3):
            self.faces.append([obj.faces[i], obj.faces[i+1], obj.faces[i+2]])

        # here is a trick, make sure vertex index same as normal index
        self.normalArray = []
        for nor in obj.normals:
            self.normalArray.append(FVector(nor[0], nor[1], nor[2]))

        self.uvArray = []
        self.colorArray = []
        self.Tangents = []  

        vertexArray = []
        for i in range(0, len(self.vertexArray)):
            vertexArray.append(FVector(self.vertexArray[i][0], self.vertexArray[i][1], self.vertexArray[i][2]))

        self.CreateMeshSection(self.CurrentMeshSectionIndex, vertexArray, self.indexArray, self.normalArray, UV0=self.uvArray, VertexColors=self.colorArray, bCreateCollision=True)
        print("*************ProceduralMesh Imported!**************")
