import sys
sys.path.append("E:\Github\SubPhysics")

import unreal_engine as ue
from unreal_engine.classes import StaticMeshComponent
from unreal_engine.classes import StaticMesh
from unreal_engine import  FVector, FRotator, FColor
from unreal_engine.classes import ProceduralMeshComponent
from unreal_engine.classes import GameplayStatics

import tensorflow as tf
from tensorflow.python import keras

import numpy as np
import ProceduralMesh
from importlib import reload

reload(ProceduralMesh)


z_path = 'E:/Github/SubPhysics/data/narray/z.npy'
w_path = 'E:/Github/SubPhysics/data/narray/w.npy'
alpha_path = 'E:/Github/SubPhysics/data/narray/alpha.npy'
beta_path = 'E:/Github/SubPhysics/data/narray/beta.npy'
x_transform_path = 'E:/Github/SubPhysics/data/narray/x_transform.npy'
y_transform_path = 'E:/Github/SubPhysics/data/narray/y_transform.npy'
x_mean_path = 'E:/Github/SubPhysics/data/narray/x_mean.npy'
y_mean_path = 'E:/Github/SubPhysics/data/narray/y_mean.npy'
model_path = 'E:/Github/SubPhysics/SavedModel/my_model.h5'
simdatacollision_path = 'E:/Github/SubPhysics/data/obj/simdatacollision.obj'
simdata_path = 'E:/Github/SubPhysics/data/obj/simdata.obj'

class DialogException(Exception):
    """
    Handy exception class for spawning a message dialog on error
    """
    def __init__(self, message):
        # 0 here, means "show only the Ok button", for other values
        # check https://docs.unrealengine.com/latest/INT/API/Runtime/Core/GenericPlatform/EAppMsgType__Type/index.html
        ue.message_dialog_open(0, message)


# Set t.maxFPS, default is 120
class Physics:

    # this is called on game start
    def begin_play(self):
        ue.log('Begin Play on class')
        print(tf.__version__)
        # set input
        player = GameplayStatics.GetPlayerController(self.uobject)
        self.uobject.EnableInput(player)
        self.uobject.bind_key('C', ue.IE_PRESSED, self.you_pressed_C)
        self.uobject.bind_key('M', ue.IE_PRESSED, self.you_pressed_M)

        self.bSimulation = True
        self.auto_move_mode = True

        # Load mesh as proceduralmesh
        self.Procedural_mesh = self.uobject.add_actor_root_component(ProceduralMesh.CustomProceduralMeshComponent, 'ProceduralMesh')
        self.Procedural_mesh.import_renderable(simdata_path)
        # self.Procedural_mesh.import_renderable('C:/Users/tongkuisu/Desktop/bunnywithnormal.obj')

        # spawn a new PyActor
        self.sphere_actor = self.uobject.actor_spawn(ue.find_class('PyActor'), FVector(0, 0, 0),FRotator(0, 0, 0))
        # add a sphere component as the root one
        self.sphere_component = self.sphere_actor.add_actor_root_component(ProceduralMesh.CustomProceduralMeshComponent, 'SphereMesh')
        # set the mesh as the Sphere asset
        self.sphere_component.import_renderable(simdatacollision_path)

        self.scale_factor = 200
        scale = self.scale_factor
        self.uobject.set_actor_scale(FVector(scale,scale,scale))
        self.uobject.set_actor_rotation(FRotator(-90, 0, 0))
        self.uobject.set_actor_location(FVector(0, 0, 0))

        self.sphere_actor.set_actor_scale(FVector(scale,scale,scale))
        self.sphere_actor.set_actor_rotation(FRotator(-90, 0, 0))
        self.sphere_actor.set_actor_location(FVector(0, 0, 0))

        # # Load model
        self.model = keras.models.load_model(model_path)
        self.model.summary()

        # select t=2, make sure t >= 2
        # input data for model [/zt;zt-1;wt]
        self.t = 0+2

        self.z = np.load(z_path)
        self.w = np.load(w_path)
        self.alpha = np.load(alpha_path)
        self.beta = np.load(beta_path)
        self.x_transform_mat = np.load(x_transform_path)
        self.x_mean = np.load(x_mean_path)
        self.y_transform_mat = np.load(y_transform_path)
        self.y_mean = np.load(y_mean_path)
        
        self.z_init_last_2 =  self.z[0, :]
        self.z_init_last_1 =  self.z[1, :]
        self.sphere_location_last = self.sphere_actor.get_actor_location()


    # this is called at every 'tick'    
    def tick(self, delta_time):
        if self.auto_move_mode:
            if self.bSimulation:
                self.auto_move_update()
        else:
            self.interactive_move_update()

    def auto_move_update(self):
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
        # print(z_init.shape)
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

    def interactive_move_update(self):
        scale = self.scale_factor
        model = self.model
        x_transform_mat = self.x_transform_mat 
        x_mean = self.x_mean
        y_mean = self.y_mean
        y_transform_mat = self.y_transform_mat 
        alpha = self.alpha 
        beta = self.beta 
        z_t_1 = self.z_init_last_1
        z_t_2 = self.z_init_last_2


        # check whether move or not
        location = self.sphere_actor.get_actor_location()
        dist_vec = location - self.sphere_location_last
        if dist_vec.x ** 2 + dist_vec.y ** 2 + dist_vec.z ** 2 < 0.1:
            return

        self.sphere_location_last = location
        real_location = np.array([location.x/scale, location.y/scale, location.z/scale]) 
        w = real_location.reshape((1,3))
        sub_w = np.matmul((w - y_mean), y_transform_mat)


        
        z_init = alpha * z_t_1 + beta * (z_t_1 - z_t_2)
        print('z_init:',z_init.shape)
        print('sub_w:',sub_w[0,:].shape)
        input_data = np.hstack((z_init, z_t_1, sub_w[0,:]))
        input_batch = np.array([input_data])
        result = model.predict(input_batch)

        #  predict 2d cause result is 2D
        predict = z_init + result
        # convert predict to real vertices(3c), got the matrix U(with u_transpose * predict)
        x_recovery = np.matmul(predict, x_transform_mat.T) + x_mean

        self.Procedural_mesh.update(x_recovery[0, :].tolist())

        self.z_init_last_2 = self.z_init_last_1
        self.z_init_last_1 = predict[0,:]
        
        print(location)
         

    # stop the simulation 
    def you_pressed_C(self):
        self.bSimulation = not self.bSimulation

    def you_pressed_M(self):
        self.auto_move_mode = not self.auto_move_mode
