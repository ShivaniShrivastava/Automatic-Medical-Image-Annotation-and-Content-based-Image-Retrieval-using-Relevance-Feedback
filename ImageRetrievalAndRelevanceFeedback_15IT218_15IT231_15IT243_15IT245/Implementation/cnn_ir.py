import tensorflow as tf # tensorflow module
import numpy as np # numpy module
import os # path join
import pandas as pd
import re

os.environ['CUDA_VISIBLE_DEVICES'] = ""


DATA_DIR = "./"
TRAINING_SET_SIZE = 11582 #25862
BATCH_SIZE = 50
TBATCH_SIZE=1
IMAGE_SIZE = 224


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# image object from protobuf
class _image_object:
    def __init__(self):
        self.image = tf.Variable([], dtype = tf.string)
        self.height = tf.Variable([], dtype = tf.int64)
        self.width = tf.Variable([], dtype = tf.int64)
        self.filename = tf.Variable([], dtype = tf.string)
        self.label = tf.Variable([], dtype = tf.int32)

## extracting information and storing them in an image object.
def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    ## TensorFlow provides several operations that you can use to convert various data formats into tensors.
    ## tf.parse_single_example() returns a dict mapping feature keys to Tensor and SparseTensor values. 
    features = tf.parse_single_example(serialized_example, features = { 
        "image/encoded": tf.FixedLenFeature([], tf.string),
        "image/height": tf.FixedLenFeature([], tf.int64),
        "image/width": tf.FixedLenFeature([], tf.int64),
        "image/filename": tf.FixedLenFeature([], tf.string),
        "image/class/label": tf.FixedLenFeature([], tf.int64),})
    image_encoded = features["image/encoded"]
    ## Decode a JPEG-encoded image to a uint8 tensor. decode(convert) images by 3-D uint8 tensors of shape [height, width, channels]
    image_raw = tf.image.decode_jpeg(image_encoded, channels=3) 
    image_object = _image_object()
    image_object.image = tf.image.resize_image_with_crop_or_pad(image_raw, IMAGE_SIZE, IMAGE_SIZE)
    image_object.height = features["image/height"]
    image_object.width = features["image/width"]
    image_object.filename = features["image/filename"]
    image_object.label = tf.cast(features["image/class/label"], tf.int64)
    return image_object

## read input images for training and testing from .tfrecord files.
def image_input(if_random = True, if_training = True):
    if(if_training):
        filenames = [os.path.join(DATA_DIR, "train-0000%d-of-00002.tfrecord" % i) for i in range(0, 1)]
    else:
        filenames = [os.path.join(DATA_DIR, "validation-0000%d-of-00002.tfrecord" % i) for i in range(0, 1)]

    for f in filenames:
        if not tf.gfile.Exists(f): ## gfile: Import router for file_io. Exists: Determines whether a path exists or not.
            raise ValueError("Failed to find file: " + f)
    ## string_input_producer: Output strings (e.g. filenames) to a queue for an input pipeline. A queue with the output strings.
    filename_queue = tf.train.string_input_producer(filenames)
    image_object = read_and_decode(filename_queue)
    image = tf.image.per_image_standardization(image_object.image)  ## Linearly scales image to have zero mean and unit norm.
#    image = image_object.image
#    image = tf.image.adjust_gamma(tf.cast(image_object.image, tf.float32), gamma=1, gain=1) # Scale image to (0, 1)
    label = image_object.label
    filename = image_object.filename

    if(if_random):
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(TRAINING_SET_SIZE * min_fraction_of_examples_in_queue)
        print("Filling queue with %d images before starting to train. " "This will take a few minutes." % min_queue_examples)
        num_preprocess_threads = 1
        image_batch, label_batch, filename_batch = tf.train.shuffle_batch( ## shuffles before adding to the queue.
            [image, label, filename],
            batch_size = BATCH_SIZE,
            num_threads = num_preprocess_threads,
            capacity = min_queue_examples + 3 * BATCH_SIZE,
            min_after_dequeue = min_queue_examples)
        return image_batch, label_batch, filename_batch
    else:
        ## tf.train.batch adds a queue to the graph to assemble a batch of examples, with possible shuffling
        image_batch, label_batch, filename_batch = tf.train.batch(
            [image, label, filename],
            batch_size = BATCH_SIZE,
            num_threads = 1)
        return image_batch, label_batch, filename_batch


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.05) ##Outputs random values from a truncated normal distribution with specified mean and std deviation. Values 2SD away from mean are dropeed 
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.02, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

## constructing the network.
def image_inference(image_batch):
    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(image_batch, [-1, IMAGE_SIZE, IMAGE_SIZE, 3])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1) # 112

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    #h_pool2 = max_pool_2x2(h_conv2) # 56

    W_conv3 = weight_variable([5, 5, 64, 128])
    b_conv3 = bias_variable([128])

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)
    #h_pool3 = max_pool_2x2(h_conv3) # 28

    W_conv4 = weight_variable([5, 5, 128, 256])
    b_conv4 = bias_variable([256])

    h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)
    #h_pool4 = max_pool_2x2(h_conv4) # 14

    # with tf.variable_scope("t1"):
    W_conv5 = weight_variable([5, 5, 256, 256])
    b_conv5 = bias_variable([256])

    h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)
    h_pool5 = max_pool_2x2(h_conv5) # 7

    with tf.variable_scope("fc"):
        W_fc1 = weight_variable([56*56*256, 2048])
        b_fc1 = bias_variable([2048])

        h_pool5_flat = tf.reshape(h_pool5, [-1, 56*56*256])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)

        h_fc1_drop = tf.nn.dropout(h_fc1, 1.0)

        W_fc2 = weight_variable([2048, 256])
        b_fc2 = bias_variable([256])

        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        W_fc3 = weight_variable([256, 64])
        b_fc3 = bias_variable([64])

        h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)

        # with tf.variable_scope("t2"):
        W_fc4 = weight_variable([64, 58])
        b_fc4 = bias_variable([58])

    y_conv = tf.nn.softmax(tf.matmul(h_fc3, W_fc4) + b_fc4) 
    # y_conv = tf.matmul(h_fc3, W_fc4) + b_fc4

    return y_conv


def image_train():
    image_batch_out, label_batch_out, filename_batch = image_input(if_random = False, if_training = True)

    image_batch_placeholder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 224, 224, 3])
    image_batch = tf.reshape(image_batch_out, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))

    label_batch_placeholder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 58])
    label_offset = -tf.ones([BATCH_SIZE], dtype=tf.int64, name="label_batch_offset") ## returns a tensor of 1's.
    
    ## Returns a one-hot tensor. First arg is the dimension of the tensor returned.
    label_batch_one_hot = tf.one_hot(tf.add(label_batch_out, label_offset), depth=58, on_value=1.0, off_value=0.0) 

    logits_out = image_inference(image_batch_placeholder)
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_batch_one_hot, logits=logits_out))
    # loss = tf.losses.mean_squared_error(labels=label_batch_placeholder, predictions=logits_out)

    # train_step = tf.train.GradientDescentOptimizer(0.007).minimize(loss) ## 0.007 is the learning rate.
    train_step=tf.train.AdamOptimizer(1e-4).minimize(loss)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Visualize the graph through tensorboard.
        file_writer = tf.summary.FileWriter("./logs", sess.graph)

        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, "C:/Users/admin/Project IAS - Copy/ImageCLEF/seperated/checkpoint-train.ckpt")
        coord = tf.train.Coordinator() ## A coordinator for threads implements a mechanism to coordinate the termination of a set of threads.
        threads = tf.train.start_queue_runners(coord=coord, sess = sess) ## Start all the queue runners collected in the graph.

        for i in range(TRAINING_SET_SIZE ):#* 100):
            image_out, label_out, label_batch_one_hot_out, filename_out = sess.run([image_batch, label_batch_out, label_batch_one_hot, filename_batch])
            # _, infer_out, loss_out = sess.run([train_step, logits_out, loss], feed_dict={image_batch_placeholder: image_out, label_batch_placeholder: label_batch_one_hot_out})
            _, infer_out, loss_out = sess.run([train_step, logits_out, loss], feed_dict={image_batch_placeholder: image_out, label_batch_placeholder: label_batch_one_hot_out})
            # _, infer_out, loss_out = sess.run([train_step, logits_out, loss], feed_dict={image_batch_placeholder: image_out, label_batch_placeholder: label_batch_one_hot_out})
            print(i)
            # a=tf.Print(t1,[t11])
            # print(a)
            # print(t21, t2)
            # print(image_out.shape)
            print("label_out: ")
            # print(filename_out)
            print(label_out)
            # print(label_batch_one_hot_out)
            print("infer_out: ")
            # print(infer_out)
            print("loss: ")
            print(loss_out)
            if(i%2 == 0): ## change to 50
                saver.save(sess, "./checkpoint-train.ckpt")

        coord.request_stop()
        coord.join(threads)
        sess.close()



def image_eval():
    image_batch_out, label_batch_out, filename_batch = image_input(if_random = False, if_training = False)

    image_batch_placeholder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 224, 224, 3])
    image_batch = tf.reshape(image_batch_out, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))

    label_tensor_placeholder = tf.placeholder(tf.int64, shape=[BATCH_SIZE])
    label_offset = -tf.ones([BATCH_SIZE], dtype=tf.int64, name="label_batch_offset") ## tf.ones create a tensor mentioned shape(1st arg) and type(2nd arg) of 1's.
    label_batch = tf.add(label_batch_out, label_offset)

    logits_out = tf.reshape(image_inference(image_batch_placeholder), [BATCH_SIZE, 58])
    logits_batch = tf.to_int64(tf.arg_max(logits_out, dimension = 1))

    correct_prediction = tf.equal(logits_batch, label_tensor_placeholder)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) ## tf.cast: Casts a tensor to a new type.
    ## tf.reduce_mean: Computes the mean of elements across dimensions of a tensor

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "./checkpoint-train.ckpt")
        coord = tf.train.Coordinator() ## A coordinator for threads implements a mechanism to coordinate the termination of a set of threads.
        threads = tf.train.start_queue_runners(coord=coord, sess = sess)

        accuracy_accu = 0.0
        pred_array=[["label","File name"]]
        pred_label=[]

        for i in range(10):
            image_out, label_out, filename_out = sess.run([image_batch, label_batch, filename_batch])

            accuracy_out, logits_batch_out,logout = sess.run([accuracy, logits_batch,logits_out], feed_dict={image_batch_placeholder: image_out, label_tensor_placeholder: label_out})
            accuracy_accu += accuracy_out

            print(i,"i")
            # print(image_out.shape)
            # print("label_out: ")
            # print(filename_out[0].decode("utf-8"))
            # print(label_out,"labels[0]")
            print(logout.shape)
            for k in range(0,BATCH_SIZE-1):
                maxi=0
                pos=0
                for l in range(0,58):  
                    if(logout[k][l]>maxi):
                        maxi=logout[k][l]
                        pos=l
                pred_label.append(pos)

            for j in range(0,BATCH_SIZE-1):
                # filename_out[j]=re.sub("[^0-9]", "", filename_out[j].decode("utf-8"))
                plabel=pred_label[j]
                filename_out[j]=filename_out[j].decode("utf-8")
                # print(j,filename_out[j])
                a=[label_out[j],filename_out[j],plabel]
                # print (a,a.shape)
                pred_array.append(a)
            # filename_out[0]=re.sub("[^0-9]", "", filename_out[0].decode("utf-8"))
            # print(filename_out,"filename_out0")
            # print(type(filename_out),"type")
            # print(label_out.shape,filename_out.shape,"shapes")
            # concat=np.concatenate((filename_out,label_out),axis=0)
            # concat=(filename_out,label_out)
            # print (concat,"concat")
            # print(pred_array)
            # print(logits_batch_out)
            print (accuracy_accu, accuracy_out)

        # print("Accuracy: ")
        # print(accuracy_accu/20 )
        df=pd.DataFrame(pred_array)
        df.to_csv("pred1.csv")
        coord.request_stop()
        coord.join(threads)
        sess.close()


def image_train2():
    # image_batch_out, label_batch_out, filename_batch = image_input(if_random = False, if_training = True)

    # image_batch_placeholder = tf.placeholder(tf.float32, shape=[TBATCH_SIZE, 224, 224, 3])
    # image_batch = tf.reshape(image_batch_out, (TBATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))

    # label_batch_placeholder = tf.placeholder(tf.float32, shape=[TBATCH_SIZE, 193])
    # label_offset = -tf.ones([TBATCH_SIZE], dtype=tf.int64, name="label_batch_offset")
    # label_batch_one_hot = tf.one_hot(tf.add(label_batch_out, label_offset), depth=193, on_value=1.0, off_value=0.0)
    # logits_out = image_inference(image_batch_placeholder)
    # # logits_out = tf.reshape(image_inference(image_batch_placeholder), [TBATCH_SIZE, 193])
    # loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=label_batch_one_hot, logits=logits_out))
    # # loss = tf.losses.mean_squared_error(labels=label_batch_placeholder, predictions=logits_out)
    # # train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="fc")
    # # train_step = optimizer.minimize(loss, var_list=train_vars)
    # # train_step = tf.train.GradientDescentOptimizer(0.007).minimize(loss, var_list=train_vars) ## 0.007 is the learning rate.
    # label_tensor_placeholder = tf.placeholder(tf.int64, shape=[BATCH_SIZE])
    # # logits_out = tf.reshape(image_inference(image_batch_placeholder), [BATCH_SIZE, 193])
    # logits_out = image_inference(image_batch_placeholder)
    # logits_batch = tf.to_int64(tf.arg_max(logits_out, dimension = 1))

    


    # train_step=tf.train.AdamOptimizer(1e-4).minimize(loss)


    # saver = tf.train.Saver()
    

    image_batch_out, label_batch_out, filename_batch = image_input(if_random = False, if_training = True)

    image_batch_placeholder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 224, 224, 3])
    image_batch = tf.reshape(image_batch_out, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))

    label_tensor_placeholder = tf.placeholder(tf.int64, shape=[BATCH_SIZE])

    label_batch_placeholder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 58])
    label_offset = -tf.ones([BATCH_SIZE], dtype=tf.int64, name="label_batch_offset") ## returns a tensor of 1's.
    
    ## Returns a one-hot tensor. First arg is the dimension of the tensor returned.
    label_batch_one_hot = tf.one_hot(tf.add(label_batch_out, label_offset), depth=58, on_value=1.0, off_value=0.0) 

    logits_out = image_inference(image_batch_placeholder)
    logits_batch = tf.to_int64(tf.arg_max(logits_out, dimension = 1))
    correct_prediction = tf.equal(logits_batch, label_tensor_placeholder)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="fc")

    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=label_batch_one_hot, logits=logits_out))
    # loss = tf.losses.mean_squared_error(labels=label_batch_placeholder, predictions=logits_out)

    # train_step = tf.train.GradientDescentOptimizer(0.007).minimize(loss) ## 0.007 is the learning rate.
    # train_step=tf.train.AdamOptimizer(1e-4).minimize(loss)
    train_step=tf.train.AdamOptimizer(1e-4).minimize(loss, var_list=train_vars)

    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        # Visualize the graph through tensorboard.
        # file_writer = tf.summary.FileWriter("./logs", sess.graph)

        sess.run(tf.global_variables_initializer())
        # print(train_vars)
        saver.restore(sess, "./checkpoint-train.ckpt")
        # W_conv4=tf.get_variable('W_conv4',[5,5,128,256])
        # W_fc3=tf.get_variable('W_fc3',[256,64])
        # print(W_conv4.eval()+"W_conv4")
        # train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="fc")

        coord = tf.train.Coordinator() ## A coordinator for threads implements a mechanism to coordinate the termination of a set of threads.
        threads = tf.train.start_queue_runners(coord=coord, sess = sess) ## Start all the queue runners collected in the graph.



        for i in range(1):#* 100):
            image_out, label_out, label_batch_one_hot_out, filename_out = sess.run([image_batch, label_batch_out, label_batch_one_hot, filename_batch])
            # t1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="fc/Variable_4:0")
            # t2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="fc/Variable:0")
            # print(label_out[2],"label_out[2]")
            # label_out[2]=93
            # print(label_out[2],"label_out[2]")
            # _, infer_out, loss_out = sess.run([train_step, logits_out, loss], feed_dict={image_batch_placeholder: image_out, label_batch_placeholder: label_batch_one_hot_out})
            infer_out = sess.run(logits_out, feed_dict={image_batch_placeholder: image_out, label_batch_placeholder: label_batch_one_hot_out})
            accuracy_out, logits_batch_out = sess.run([accuracy, logits_batch], feed_dict={image_batch_placeholder: image_out, label_tensor_placeholder: label_out})

            print(i)
            # print(Wcov,Wfc)
            # print (train_vars)
            # print(image_out.shape)
            print("label_out: ")
            print(filename_out)
            print(infer_out,"Infer out before\n")
            print(logits_batch_out,"logits_batch_out\n")
            # reply=int(raw_input("Any feedback  0: No    1:Yes  "))
            reply=1
            if(reply==1):
                label_batch_out= [55]#int(raw_input("Enter new label"))
                # label_batch_out.assign([55,]).eval()
                print(type(label_batch_out),"labels_batch_out\n")
                label_batch_one_hot=tf.one_hot(label_batch_out, depth=58, on_value=1.0, off_value=0.0)
                _, infer_out1, loss_out = sess.run([train_step, logits_out, loss], feed_dict={image_batch_placeholder: image_out, label_batch_placeholder: label_batch_one_hot_out})
                infer_out2 = sess.run(logits_out, feed_dict={image_batch_placeholder: image_out, label_batch_placeholder: label_batch_one_hot_out})
            print(infer_out1,"Infer out\n")
            print(infer_out2,"Infer out 2\n")
            print(np.array_equal(infer_out,infer_out1))
            print(np.array_equal(infer_out,infer_out2))
            # assert infer_out == infer_out1
            # print(label_batch_one_hot_out)
            # print("infer_out: ")
            # print(infer_out)
            print("loss: ")
            print(loss_out)
            # if(i%2 == 0): ## change to 50
            #     saver.save(sess, "C:/Users/admin/Project IAS - Copy/ImageCLEF/seperated/checkpoint-train.ckpt")

        coord.request_stop()
        coord.join(threads)
        sess.close()


# image_train()
image_eval()
# image_train2()