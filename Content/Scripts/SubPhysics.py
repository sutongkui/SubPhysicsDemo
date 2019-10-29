import sys
sys.path.append("E:\Github\SubPhysics")


import unreal_engine as ue
from unreal_engine.classes import StaticMeshComponent
from unreal_engine.classes import StaticMesh
from unreal_engine import  FVector, FRotator
from unreal_engine.classes import ProceduralMeshComponent

import tensorflow as tf
from tensorflow.python import keras

import numpy as np

z_path = 'E:/Github/SubPhysics/data/narray/z.npy'
w_path = 'E:/Github/SubPhysics/data/narray/w.npy'
alpha_path = 'E:/Github/SubPhysics/data/narray/alpha.npy'
beta_path = 'E:/Github/SubPhysics/data/narray/beta.npy'
x_transform_path = 'E:/Github/SubPhysics/data/narray/x_transform.npy'
y_transform_path = 'E:/Github/SubPhysics/data/narray/y_transform.npy'
x_mean_path = 'E:/Github/SubPhysics/data/narray/x_mean.npy'
y_mean_path = 'E:/Github/SubPhysics/data/narray/y_mean.npy'

class DialogException(Exception):
    """
    Handy exception class for spawning a message dialog on error
    """
    def __init__(self, message):
        # 0 here, means "show only the Ok button", for other values
        # check https://docs.unrealengine.com/latest/INT/API/Runtime/Core/GenericPlatform/EAppMsgType__Type/index.html
        ue.message_dialog_open(0, message)
# If we separate this to two file, and it seems that ue can't get the changes from another file
# only handle vertex and index buffer
class OBJ:
    def __init__(self, filename, swapyz=False):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.faces = []

        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = [float(values[1]), float(values[2]), float(values[3])]
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append([v[0], v[1], v[2]])
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    self.faces.append(int(w[0])-1)
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)

        print("Load obj vertices***")
        print(len(self.vertices))


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

    def update(self, vertices):
        if len(self.vertexArray) != len(vertices) and  len(self.vertexArray)*3 != len(vertices):
            print("***** update vertices number not equal to original************")
            return

        uvArray = self.uvArray
        colorArray = self.colorArray
        normalArray = self.normalArray

        # optimize this later
        vertexArray = []
        for i in range(0, len(vertices), 3):
            vertexArray.append(FVector(vertices[i], vertices[i+1], vertices[i+2]))


        self.UpdateMeshSection(self.CurrentMeshSectionIndex, vertexArray, normalArray, UV0=uvArray, VertexColors=colorArray)

    def import_renderable(self, filename):
        """Adds the specified renderable as a mesh section to this procedural mesh component"""
        # obj = ObjLoader.OBJ(filename)
        obj = OBJ(filename)

        self.vertexArray = []
        for ver in obj.vertices:
            self.vertexArray.append(FVector(ver[0], ver[1], ver[2]))
        
        self.indexArray = obj.faces

        self.normalArray = []
        self.uvArray = []
        self.colorArray = []
        self.Tangents = []  

        self.CreateMeshSection(self.CurrentMeshSectionIndex, self.vertexArray, self.indexArray, self.normalArray, UV0=self.uvArray, VertexColors=self.colorArray, bCreateCollision=True)
        print("*************ProceduralMesh Imported!**************")

# Set t.maxFPS, default is 120
class Physics:

    # this is called on game start
    def begin_play(self):
        ue.log('Begin Play on class')
        print(tf.__version__)

        #test = self.uobject.actor_create_default_subobject(ProceduralMeshComponent, "Test")

        # Load mesh as proceduralmesh
        self.Procedural_mesh = self.uobject.add_actor_root_component(CustomProceduralMeshComponent, 'ProceduralMesh')
        self.Procedural_mesh.import_renderable('E:/Github/SubPhysics/data/obj/simdata.obj')


        # self.sphere = self.uobject.add_actor_component(CustomProceduralMeshComponent, 'Sphere')
        # self.sphere.import_renderable('E:/Github/SubPhysics/data/obj/simdatacollision.obj')

        # spawn a new PyActor
        self.sphere_actor = self.uobject.actor_spawn(ue.find_class('PyActor'), FVector(0, 0, 0),FRotator(0, 0, 0))
        # add a sphere component as the root one
        self.sphere_component = self.sphere_actor.add_actor_root_component(ue.find_class('CustomProceduralMeshComponent'), 'SphereMesh')
        # set the mesh as the Sphere asset
        self.sphere_component.import_renderable('E:/Github/SubPhysics/data/obj/simdatacollision.obj')

        self.scale_factor = 200
        scale = self.scale_factor
        self.uobject.set_actor_scale(FVector(scale,scale,scale))
        self.uobject.set_actor_rotation(FRotator(-90, 0, 0))

        self.sphere_actor.set_actor_scale(FVector(scale,scale,scale))
        self.sphere_actor.set_actor_rotation(FRotator(-90, 0, 0))

        # Load model
        self.model = keras.models.load_model('E:/Github/SubPhysics/SavedModel/my_model.h5')
        self.model.summary()

        # select t=2, make sure t >= 2
        # input data for model [/zt;zt-1;wt]
        self.t = 1+2

        self.z = np.load(z_path)
        self.w = np.load(w_path)
        self.alpha = np.load(alpha_path)
        self.beta = np.load(beta_path)
        self.x_transform_mat = np.load(x_transform_path)
        self.x_mean = np.load(x_mean_path)
        self.y_transform_mat = np.load(y_transform_path)
        self.y_mean = np.load(y_mean_path)

    # this is called at every 'tick'    
    def tick(self, delta_time):


        model = self.model
        t = self.t
        z = self.z
        w = self.w
        alpha = self.alpha 
        beta = self.beta 
        x_transform_mat = self.x_transform_mat 
        x_mean = self.x_mean
        y_transform_mat = self.y_transform_mat 
        y_mean = self.y_mean


        z_init = alpha * z[t-1, :] + beta * (z[t-1, :] - z[t-2, :])
        input_data = np.hstack((z_init, z[t-1, :], w[t, :]))
        input_batch = np.array([input_data])
        result = model.predict(input_batch)


        predict = z_init + result
        # convert predict to real vertices(3c), got the matrix U(with u_transpose * predict)
        x_recovery = np.matmul(predict, x_transform_mat.T) + x_mean

        self.Procedural_mesh.update(x_recovery[0, :].tolist())

        y_recovery = np.matmul(w[t, :], y_transform_mat.T) + y_mean
        y_list = y_recovery.tolist()
        scale = self.scale_factor
        self.sphere_actor.set_actor_location(FVector(y_list[0]*scale, y_list[1]*scale, y_list[2]*scale))

        self.t += 1
        a = 1
