import numpy as np
from sklearn.neighbors import KDTree
import open3d as o3d

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import importlib


def pc_normalize(pc):
    centroid = np.mean(pc[:,:3], axis=0)
    pc[:,:3] = pc[:,:3] - centroid
    m = np.max(np.sqrt(np.sum(pc[:,:3]**2, axis=1)))
    pc[:,:3] = pc[:,:3] / m
    return pc


def placeholder_inputs(batch_size, num_point):
    pointclouds_ph = tf.placeholder(tf.float32, shape=(batch_size, num_point, 6))
    input_label_ph = tf.placeholder(tf.float32, shape=(batch_size, 2))
    return pointclouds_ph, input_label_ph


class pointNetSeg:
    def __init__(self, num_obj_cats, num_part_cats, obj_idx, batch_size, num_point, gpu_index, model_name, stored_model_path, naming_scope):
        self.is_training = False

        labels = [0,0]

        labels[obj_idx] = 1
        self.cur_labels_one_hot = np.array( [ labels ] * batch_size )

        # load the model name
        model = importlib.import_module(model_name)

        # Enables the system to run on CPU
        if( int(gpu_index) < 0 ):
            gpu_device = '/cpu:0'
        else:
            gpu_device = '/gpu:'+str(gpu_index)

        with tf.device(gpu_device):
            self.pointclouds_ph, self.input_label_ph = placeholder_inputs(batch_size, num_point)
            self.is_training_ph = tf.placeholder(tf.bool, shape=())
            with tf.variable_scope(naming_scope):
                self.cat_pred, end_points = model.get_model(self.pointclouds_ph, \
                      is_training=self.is_training_ph, cat_num=num_obj_cats, \
                      part_num=NUM_PART_CATS, batch_size=batch_size, num_point=num_point, weight_decay=0)
                
                
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        self.sess = tf.Session(config=config)

        saver.restore(self.sess, stored_model_path)

    def predict(self, cur_data_pcd):
        cat_pred_res = self.sess.run(
                [self.cat_pred], feed_dict={
                self.pointclouds_ph: cur_data_pcd,
                self.is_training_ph: self.is_training,
                })
        return cat_pred_res



### Setup network
NUM_CATEGORIES = 2
NUM_PART_CATS = 2
batch_size = 1
max_num_point = 1024*8

gpu_index = -1 #cpu

model_name = 'pin_seg_model'
model_path = 'trained_models_o3d/epoch_200.ckpt'
scope = "pin_seg"

pointNet = pointNetSeg(NUM_CATEGORIES, NUM_PART_CATS, 1, batch_size, max_num_point, gpu_index, model_name, model_path, scope)



### Load object
#t = '../../danchell/additional_content/database/cad/cap2-6.ply'


t1 = 'data/78X5071_mm_rot.ply'
t2 = 'data/1003232_cut_rot.ply'
t3 = 'data/1162740_mm.ply'
t4 = 'data/1668591_mm.ply'
t5 = 'data/1673308_mm.ply'
t6 = 'data/7429312.ply'
t7 = 'data/cap2-6_remesh.ply'

t_list = [t1, t2, t3, t4, t5, t6, t7]

#t = '/home/frhag/workspace/danchell/additional_content/cad_model_generation/my_model_production.ply'
#t = 'plastic_out_new_128.ply'

for t in t_list:

    # out_name = t.split('/')[-1]
    
    out_name = t

    print( out_name )

    textured_mesh = o3d.io.read_triangle_mesh( t )

    pcd_o3d = textured_mesh.sample_points_uniformly(number_of_points=max_num_point)
    pcd_o3d.estimate_normals()
    
    # o3d.visualization.draw_geometries([ pcd_o3d ])

    pointcloud_pointnet = np.concatenate( [np.asarray(pcd_o3d.points), np.asarray(pcd_o3d.normals)], axis = 1 )
    pointcloud_pointnet = pc_normalize(pointcloud_pointnet)

    # Run through the network
    label_pred_val = pointNet.predict([pointcloud_pointnet])

    cat_segmentation = np.argmax(label_pred_val[0], axis=2)

    colors = [[1,0,0],[0,1,0]]

    new_colors = []
    for col_index in range(max_num_point):
        new_colors.append( colors[cat_segmentation[0,col_index]] )
    pcd_o3d.colors = o3d.utility.Vector3dVector(np.array(new_colors))

    # o3d.visualization.draw_geometries([ pcd_o3d ])

    ### Project to cad
    vertexes = np.asarray(textured_mesh.vertices)
    vertex_colors = np.asarray(textured_mesh.vertex_colors)

    tree = KDTree( np.asarray(pcd_o3d.points) , leaf_size=2)
    dist_feature_list, index_feature_list = tree.query(vertexes , k=15) # Instead of only taking nearest, 15 points are used.

    colors = [[0.5,0.5,0.5],[1,0,0]]

    for vertex_id in range(len(vertexes)):
        mean_index = []
        for index in index_feature_list[vertex_id]:
            mean_index.append( cat_segmentation[0, index]  )
        mean_index = int(np.round(np.mean(mean_index)))
        vertex_colors[vertex_id] = colors[mean_index]

    textured_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    # o3d.visualization.draw_geometries([ textured_mesh ])

    # o3d.io.write_triangle_mesh("net_mesh.ply", textured_mesh)
    o3d.io.write_triangle_mesh(out_name, textured_mesh)
