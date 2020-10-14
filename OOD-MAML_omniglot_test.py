import tensorflow as tf

from gen_meta_omni import Gen_data_Meta
import numpy as np
import os
from tensorflow.contrib.layers.python import layers as tf_layers



data_folder='directory of omniglot data'
Model_save_folder='directory of tf model save file'
saver = tf.train.Saver()
sess=tf.Session()
saver.restore(sess,"Model_save_folder")

# 5-SHOT 5-WAY
ways=5
num_in=5
num_in_and_out=2*ways*num_in
dim_hidden=[64,64,64,64]
num_fake_img = 3  # number of fake image (M)

update_lr=1e-1

gen_class=Gen_data_Meta(data_folder=data_folder,num_in=num_in,mode='test',num_in_and_out=num_in_and_out)

X_in=tf.placeholder('float',[ways*num_in, 28,28,1])
Y_in=tf.placeholder('float',[ways*num_in,ways])
X_in_and_out=tf.placeholder('float',[num_in_and_out, 28,28,1])
Y_in_and_out=tf.placeholder('float',[num_in_and_out,ways])

def construct_conv_weights():
    initializer_w=tf.contrib.layers.xavier_initializer_conv2d()
    initializer_b=tf.contrib.layers.xavier_initializer()

    par_real1={}

    with tf.variable_scope("par_real",reuse=tf.AUTO_REUSE):
        shape_w=[3,3,1,dim_hidden[0]]
        shape_b=[dim_hidden[0]]

        par_real1['w1'] = tf.Variable(initializer_w(shape=shape_w),name='w1')
        par_real1['b1'] = tf.Variable(initializer_b(shape=shape_b),name='b1')
        for i in range(1, len(dim_hidden)):
            shape_w=[3,3,dim_hidden[i - 1], dim_hidden[i]]
            shape_b=[dim_hidden[i]]

            par_real1['w' + str(i + 1)] = tf.Variable(initializer_w(shape=shape_w), name='w' + str(i + 1))
            par_real1['b' + str(i + 1)] = tf.Variable(initializer_b(shape=shape_b), name='b' + str(i + 1))
    with tf.variable_scope("par_fake", reuse=tf.AUTO_REUSE):
        initializer_fake = tf.random_normal_initializer()
        initializer_alpha = tf.ones_initializer()
        fake_img_shape = [num_fake_img, 28,28, 1]
        fake_img_par = tf.Variable(initializer_fake(shape=fake_img_shape), name='fake_img')
        beta_fake = tf.Variable(initializer_alpha(shape=fake_img_shape), name='alpha')
    return par_real1,fake_img_par,beta_fake
par_real1,fake_img,beta_fake=construct_conv_weights()  # theta, theta_{fake}, beta_fake
with tf.variable_scope('par_real/', reuse=tf.AUTO_REUSE):
    dim_w1 = [64*4, 2]
    dim_b1 = [2]
    initializer_fc=tf.contrib.layers.xavier_initializer()
    par_real1['w_fc1']=tf.Variable(name='w_fc1',initial_value=initializer_fc(shape=dim_w1))
    par_real1['b_fc1']=tf.Variable(name='b_fc1',initial_value=initializer_fc(shape=dim_b1))

def conv_block(input,cweight,bweight,scope,reuse=False):
    stride,no_stride=[1,2,2,1],[1,1,1,1]
    conv_output=tf.nn.conv2d(input,cweight,strides=stride,padding='SAME')+bweight
    normed=bat_normalize(conv_output,activation=tf.nn.elu,scope=scope,reuse=reuse)

    return normed

def bat_normalize(input,activation,scope,reuse=False):
    h=tf_layers.batch_norm(input,activation_fn=activation,reuse=reuse,scope=scope)

    return h

def conv2d(input,par,reuse=False):
    h_past=input
    for i in range(len(dim_hidden)):
        h_curr=conv_block(h_past,par['w'+str(i+1)],par['b'+str(i+1)],scope=str(i+1),reuse=reuse)
        h_past=h_curr
    return h_past

def fc_class(input,par,reuse=True):
    input_fc = tf.layers.flatten(input)
    result = tf.matmul(input_fc, par['w_fc1']) + par['b_fc1']
    return result


def loss_softmax(input, par, input_fake=None, reuse=True, label=None):
    repr = conv2d(input, par, reuse=reuse)
    input_loss = (fc_class(repr, par, reuse=reuse))

    if label == None:
        input_shape = input.shape.as_list()
        input_fake_shape = input_fake.shape.as_list()

        repr_fake = conv2d(tf.sigmoid(input_fake), par, reuse=reuse)
        fake_loss = (fc_class(repr_fake, par, reuse=True))

        ones_fake = tf.ones([input_fake_shape[0], 1])
        zeros_fake = tf.zeros([input_fake_shape[0], 1])
        label_fake = tf.concat((ones_fake, zeros_fake), axis=1)

        zeros = tf.zeros([input_shape[0], 1])
        ones = tf.ones([input_shape[0], 1])
        label_real = tf.concat((zeros, ones), axis=1)

        result_img = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_real, logits=input_loss))
        result_fake = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_fake, logits=fake_loss))
        result = result_img + result_fake
    else:

        label_re = tf.reshape(label, [-1, 1])
        label_real = tf.concat((label_re, 1.0 - label_re), axis=1)
        result = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_real, logits=input_loss))

    return result

def loss_adv(input_fake, par):
    repr_fake = conv2d(tf.sigmoid(input_fake), par, reuse=True)
    fake_loss = (fc_class(repr_fake, par, reuse=True))



    fake_shape = input_fake.shape.as_list()
    ones_fake = tf.ones([fake_shape[0], 1])
    zeros_fake = tf.zeros([fake_shape[0], 1])
    label_fake = tf.concat((zeros_fake, ones_fake), axis=1)

    result_fake = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_fake, logits=fake_loss))
    return result_fake
def grad_update_par(input, par, input_fake, reuse=True, stop_crit=True,num_iter=5):
    loss_real = loss_softmax(input=input, input_fake=input_fake, par=par, reuse=reuse)
    grad_real = tf.gradients(loss_real, list(par.values()))
    if stop_crit:
        grad_real = [tf.stop_gradient(grad) for grad in grad_real]
    gradient_real = dict(zip(par.keys(), grad_real))
    par_real_update = dict(
        zip(par.keys(), [par[key] - update_lr * gradient_real[key] for key in par.keys()]))

    for k in range(num_iter-1):
        loss_real = loss_softmax(input=input, input_fake=input_fake, par=par_real_update, reuse=True)
        grad_real = tf.gradients(loss_real, list(par_real_update.values()))
        if stop_crit:
            grad_real = [tf.stop_gradient(grad) for grad in grad_real]
        gradient_real = dict(zip(par_real_update.keys(), grad_real))
        par_real_update = dict(
            zip(par_real_update.keys(),
                [par_real_update[key] - update_lr * gradient_real[key] for key in par_real_update.keys()]))

    return par_real_update
def grad_update_fake_img(input_fake,grad_beta, par):
    loss_real_adv = loss_adv(input_fake=input_fake,  par=par)
    grad_real_adv = tf.gradients(loss_real_adv, [input_fake])
    grad_real_adv = tf.stop_gradient(grad_real_adv)
    input_fake_update = input_fake - tf.multiply(tf.nn.softplus(grad_beta) , tf.sign(grad_real_adv[0]))

    for k in range(2):
        loss_real_adv = loss_adv(input_fake=input_fake_update,par=par)
        grad_real_adv = tf.gradients(loss_real_adv, [input_fake_update])
        grad_real_adv = tf.stop_gradient(grad_real_adv)
        input_fake_update = input_fake_update - tf.multiply(tf.nn.softplus(grad_beta) , tf.sign(grad_real_adv[0]))

    loss_adv_final = loss_adv(input_fake=input_fake_update,par=par)
    return input_fake_update, loss_adv_final

def weight_real_update(input, par, input_fake,grad_beta,  reuse=True):
    par_real_update = grad_update_par(input=input, par=par, reuse=reuse, input_fake=input_fake,stop_crit=False,num_iter=3)
    input_fake_update, loss_adv_final = grad_update_fake_img(input_fake=input_fake, par=par_real_update, grad_beta=grad_beta)

    fake_stack = tf.concat((input_fake, input_fake_update), axis=0)

    par_real_update_v2 = grad_update_par(input=input, par=par, input_fake=fake_stack, stop_crit=False,
                                      reuse=True,num_iter=5)
    #######################################################################################################################

    return par_real_update_v2, loss_adv_final,input_fake_update


unused=conv2d(X_in,par=par_real1,reuse=False)

Eye_label=tf.eye(ways)

def sub_task(inp):
    Eye_label_i=inp
    Eye_label_i=tf.reshape(Eye_label_i,[1,-1])
    io_idx=tf.equal(tf.reduce_sum(tf.abs(Y_in-Eye_label_i),axis=1),0.0)
    #ix_idx = tf.not_equal(tf.reduce_sum(tf.abs(Y_in - Eye_label_i), axis=1), 0.0)

    X_in_i=tf.boolean_mask(X_in,io_idx)

    par_real_update_i, _, _ = weight_real_update(input=X_in_i, par=par_real1, input_fake=fake_img, grad_beta=beta_fake,
                                                 reuse=True)

    repr_tar_i = conv2d(X_in_and_out, par_real_update_i, reuse=True)
    c_tar_i = tf.nn.softmax(fc_class(repr_tar_i, par_real_update_i, reuse=True))
    return c_tar_i

elems = (Eye_label)
out_dtype = (tf.float32)
result_tot = tf.map_fn(sub_task, elems=elems, dtype=out_dtype,parallel_iterations=ways)  #[5,50,2]

result=result_tot

X_in1, Y_in1, X_in_and_out1, Y_in_and_out1, Y_in2 = gen_class.construction_unknown_single_test(ways)
X_in_feed=np.expand_dims(X_in1,-1)
X_in_and_out_feed=np.expand_dims(X_in_and_out1,-1)
Y_in_feed=np.eye(ways)[Y_in1]
feed_dict={X_in:X_in_feed,Y_in:Y_in_feed,X_in_and_out:X_in_and_out_feed}

OOD_detect_binary=sess.run(result,feed_dict=feed_dict)

OOD_detect=np.transpose(OOD_detect_binary[:,:,0])
OOD_detect_result=1.0*(np.min(OOD_detect,axis=1)>0.5)
OOD_detect_filter=np.argmin(OOD_detect_result,axis=1)

class_result_stat= np.reshape(np.sum(Y_in2==OOD_detect_filter[:num_in*ways])/(num_in*ways),[1])
OOD_result_stat= np.reshape(np.sum(Y_in_and_out1==OOD_detect_result)/(num_in_and_out),[1])
tnr_result_stat= np.reshape(np.sum((Y_in_and_out1==OOD_detect_result)[num_in*ways:num_in_and_out])/(num_in*ways),[1])


print (class_result_stat,OOD_result_stat,tnr_result_stat)  #[classification, detection accuracy, tnr accuracy]







