import tensorflow as tf
import numpy as np
import math
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../models'))
import tf_util


def knn_range(adj_matrix, k=20, search_range=20):
  """Get KNN based on the pairwise distance.
  Args:
    pairwise distance: (batch_size, num_points, num_points)
    k: int
    search_range: int

  Returns:
    nearest neighbors: (batch_size, num_points, k)
  """
  neg_adj = -adj_matrix
  values, nn_idx = tf.nn.top_k(neg_adj, k=search_range)
  nn_idx = tf.transpose(nn_idx)
  nn_idx = tf.random.shuffle(nn_idx)
  nn_idx = tf.transpose(nn_idx)
  return nn_idx[:,:,:k], values

def knn_rand(adj_matrix, k=20):
  """Get KNN based on the pairwise distance.
  Args:
    pairwise distance: (batch_size, num_points, num_points)
    k: int
  Returns:
    nearest neighbors: (batch_size, num_points, k)
  """
  neg_adj = -adj_matrix
  _, nn_idx = tf.nn.top_k(neg_adj, k=(k*3))
  nn_idx = tf.transpose(nn_idx)
  nn_idx = tf.random.shuffle(nn_idx)
  nn_idx = tf.transpose(nn_idx)
  return nn_idx[:,:,:k]

def get_transform(point_cloud, is_training, bn_decay=None, K = 6):
    """ Transform Net, input is BxNx3 gray image
        Return:
            Transformation matrix of size 3xK """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value

    input_image = tf.expand_dims(point_cloud, -1)
    net = tf_util.conv2d(input_image, 64, [1,6], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='tconv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='tconv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='tconv4', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point,1], padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 128, bn=True, is_training=is_training, scope='tfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 128, bn=True, is_training=is_training, scope='tfc2', bn_decay=bn_decay)

    with tf.variable_scope('transform_XYZ') as sc:
        assert(K==6)
        weights = tf.get_variable('weights', [128, K*K], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        biases = tf.get_variable('biases', [K*K], initializer=tf.constant_initializer(0.0), dtype=tf.float32) + tf.constant(np.eye(K).flatten(), dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, K, K])
    return transform




def input_transform_net(edge_feature, is_training, bn_decay=None, K=6, is_dist=False):
  """ Input (XYZ) Transform Net, input is BxNx3 gray image
    Return:
      Transformation matrix of size 3xK """
  batch_size = edge_feature.get_shape()[0].value
  num_point = edge_feature.get_shape()[1].value

  net = tf_util.conv2d(edge_feature, 64, [1,1],
             padding='VALID', stride=[1,1],
             bn=True, is_training=is_training,
             scope='tconv1', bn_decay=bn_decay)
  net = tf_util.conv2d(net, 128, [1,1],
             padding='VALID', stride=[1,1],
             bn=True, is_training=is_training,
             scope='tconv2', bn_decay=bn_decay)
  
  net = tf.reduce_max(net, axis=-2, keep_dims=True)
  
  net = tf_util.conv2d(net, 1024, [1,1],
             padding='VALID', stride=[1,1],
             bn=True, is_training=is_training,
             scope='tconv3', bn_decay=bn_decay)
  net = tf_util.max_pool2d(net, [num_point,1],
               padding='VALID', scope='tmaxpool')

  net = tf.reshape(net, [batch_size, -1])
  net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                  scope='tfc1', bn_decay=bn_decay)
  net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                  scope='tfc2', bn_decay=bn_decay)

  with tf.variable_scope('transform_XYZ') as sc:
    with tf.device('/cpu:0'):
      weights = tf.get_variable('weights', [256, K*K],
                    initializer=tf.constant_initializer(0.0),
                    dtype=tf.float32)
      biases = tf.get_variable('biases', [K*K],
                   initializer=tf.constant_initializer(0.0),
                   dtype=tf.float32)
    biases += tf.constant(np.eye(K).flatten(), dtype=tf.float32)
    transform = tf.matmul(net, weights)
    transform = tf.nn.bias_add(transform, biases)

  transform = tf.reshape(transform, [batch_size, K, K])
  return transform


def get_model(point_cloud, is_training, cat_num, part_num, \
		batch_size, num_point, weight_decay, bn_decay=None):
  """ ConvNet baseline, input is BxNx3 gray image """
  end_points = {}

  batch_size = point_cloud.get_shape()[0].value
  num_point = point_cloud.get_shape()[1].value
  
  k = 10

  with tf.variable_scope('pro/transform_net1') as sc:
    transform = get_transform(point_cloud, is_training, bn_decay, K=6)

    
  with tf.variable_scope('pro/mul_1') as sc:
    point_cloud_transformed = tf.matmul(point_cloud, transform)
  
  end_points['transform'] = transform
      
  with tf.variable_scope('pro/edge_2') as sc:
      input_image = tf.expand_dims(point_cloud_transformed, -1)
      adj = tf_util.pairwise_distance(point_cloud_transformed[:,:,:])

      nn_idx, values = knn_range(adj, k=k, search_range=k*3)

      edge_feature_2 = tf_util.get_edge_feature(input_image, nn_idx=nn_idx, k=k)#, scope='pro/edge2')

  out1 = tf_util.conv2d(edge_feature_2, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='pro/adj_conv1', bn_decay=bn_decay)
  
  out2 = tf_util.conv2d(out1, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='pro/adj_conv2', bn_decay=bn_decay)

  with tf.variable_scope('pro/red_1') as sc:
    net_1 = tf.reduce_max(out2, axis=-2, keep_dims=True) # , scope='pro/red1'


  with tf.variable_scope('pro/edge_3') as sc:

      values_mean_3 = tf.reduce_mean(values[:,:,-1])
      adj = adj + (tf.round(tf.nn.sigmoid(-adj-(2*values_mean_3)))*100000000)      
      nn_idx = knn_rand(adj, k=k) 
      
      edge_feature_3 = tf_util.get_edge_feature(net_1, nn_idx=nn_idx, k=k)

  out3 = tf_util.conv2d(edge_feature_3, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='pro/adj_conv3', bn_decay=bn_decay)

  out4 = tf_util.conv2d(out3, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='pro/adj_conv4', bn_decay=bn_decay)

  with tf.variable_scope('pro/red_2') as sc:
    net_2 = tf.reduce_max(out4, axis=-2, keep_dims=True)
  
  
  with tf.variable_scope('pro/edge_4') as sc:
      adj = adj + (tf.round(tf.nn.sigmoid(-adj-(4*values_mean_3)))*100000000)      
      nn_idx = knn_rand(adj, k=k)

      edge_feature_4 = tf_util.get_edge_feature(net_2, nn_idx=nn_idx, k=k)




  out5 = tf_util.conv2d(edge_feature_4, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='pro/adj_conv5', bn_decay=bn_decay)

  with tf.variable_scope('pro/red_3') as sc:
    net_3 = tf.reduce_max(out5, axis=-2, keep_dims=True)



  out7 = tf_util.conv2d(tf.concat([net_1, net_2, net_3], axis=-1, name='pro/concat_mean'), 1024, [1, 1], 
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='pro/adj_conv7', bn_decay=bn_decay, activation_fn=tf.nn.elu)
                       
  out_mean = tf_util.avg_pool2d(out7, [num_point, 1], padding='VALID', scope='pro/meanpool') # proc/

  expand_mean = tf.tile(out_mean, [1, num_point, 1, 1], name='pro/expand_mean')
  
  out7_max = tf_util.conv2d(tf.concat([net_1, net_2, net_3, expand_mean], axis=-1, name='pro/concat_max'), 1024, [1, 1], 
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='pro/adj_conv7_max', bn_decay=bn_decay, activation_fn=tf.nn.elu)
  out_max = tf_util.max_pool2d(out7_max, [num_point, 1], padding='VALID', scope='pro/maxpool')

  expand_max = tf.tile(out_max, [1, num_point, 1, 1], name='pro/expand_max')

  concat = tf.concat(axis=3, values=[expand_mean,
                                     expand_max, 
                                     net_1,
                                     net_2,
                                     net_3],
                                     name='seg/concat')

  net2 = tf_util.conv2d(concat, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
            bn=True, is_training=is_training, scope='seg/conv1', weight_decay=weight_decay)
  net2 = tf_util.dropout(net2, keep_prob=0.6, is_training=is_training, scope='seg/dp1')
  net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
            bn=True, is_training=is_training, scope='seg/conv2', weight_decay=weight_decay)
  net2 = tf_util.dropout(net2, keep_prob=0.6, is_training=is_training, scope='seg/dp2')
  net2 = tf_util.conv2d(net2, 128, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
            bn=True, is_training=is_training, scope='seg/conv3', weight_decay=weight_decay)
            
  category = tf_util.conv2d(net2, cat_num, [1,1], padding='VALID', stride=[1,1], activation_fn=None, 
            bn=False, scope='seg_cat/conv4', weight_decay=weight_decay)
  category = tf.reshape(category, [batch_size, num_point, cat_num])
            

  return category, end_points
  

def get_loss(cat_pred, cat, end_points):
    
    per_instance_cat_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cat_pred, labels=cat), axis=1)
    ##per_instance_seg_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=seg_pred, labels=seg), axis=1)
    cat_loss = tf.reduce_mean(per_instance_cat_loss)

    per_instance_cat_pred_res = tf.argmax(cat_pred, 2)
    
    # Enforce the transformation as orthogonal matrix
    transform = end_points['transform'] # BxKxK
    K = transform.get_shape()[1].value
    mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0,2,1])) - tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff) 
    

    total_loss = cat_loss + mat_diff_loss * 1e-3


    return total_loss, cat_loss, per_instance_cat_loss, per_instance_cat_pred_res



def pairwise_distance(point_cloud):
  """Compute pairwise distance of a point cloud.

  Args:
    point_cloud: tensor (batch_size, num_points, num_dims)

  Returns:
    pairwise distance: (batch_size, num_points, num_points)
  """
  og_batch_size = point_cloud.get_shape().as_list()[0]
  point_cloud = tf.squeeze(point_cloud)
  if og_batch_size == 1:
    point_cloud = tf.expand_dims(point_cloud, 0)
    
  point_cloud_transpose = tf.transpose(point_cloud, perm=[0, 2, 1])
  point_cloud_inner = tf.matmul(point_cloud, point_cloud_transpose)
  point_cloud_inner = -2*point_cloud_inner
  point_cloud_square = tf.reduce_sum(tf.square(point_cloud), axis=-1, keep_dims=True)
  point_cloud_square_tranpose = tf.transpose(point_cloud_square, perm=[0, 2, 1])
  return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose


def knn(adj_matrix, k=20):
  """Get KNN based on the pairwise distance.
  Args:
    pairwise distance: (batch_size, num_points, num_points)
    k: int

  Returns:
    nearest neighbors: (batch_size, num_points, k)
  """
  neg_adj = -adj_matrix
  _, nn_idx = tf.nn.top_k(neg_adj, k=k)
  return nn_idx


def get_edge_feature(point_cloud, nn_idx, k=20):
  """Construct edge feature for each point
  Args:
    point_cloud: (batch_size, num_points, 1, num_dims)
    nn_idx: (batch_size, num_points, k)
    k: int

  Returns:
    edge features: (batch_size, num_points, k, num_dims)
  """
  og_batch_size = point_cloud.get_shape().as_list()[0]
  point_cloud = tf.squeeze(point_cloud)
  if og_batch_size == 1:
    point_cloud = tf.expand_dims(point_cloud, 0)

  point_cloud_central = point_cloud

  point_cloud_shape = point_cloud.get_shape()
  batch_size = point_cloud_shape[0].value
  num_points = point_cloud_shape[1].value
  num_dims = point_cloud_shape[2].value

  idx_ = tf.range(batch_size) * num_points
  idx_ = tf.reshape(idx_, [batch_size, 1, 1]) 

  point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
  point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx+idx_)
  point_cloud_central = tf.expand_dims(point_cloud_central, axis=-2)

  point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])

  edge_feature = tf.concat([point_cloud_central, point_cloud_neighbors-point_cloud_central], axis=-1)
  return edge_feature

