#References: https://github.com/bolducp/hierarchical-rnn/tree/master/hmlstm
from matplotlib import pyplot as plt

import HMSLSTM_Main as Main
import tensorflow as tf
import numpy as np
# from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
print(tf.__version__)





def regression_per_video(labels, predictions, starts_list,
                         idx):  # starts_list includes the index of the start index in each video in the main input file
    #VRE
    labels_pool = np.array([0, 5, 10])
    np.clip(predictions, 0, 10, out=predictions)
    LOSS = 0
    for i, start in enumerate(starts_list):
        if (i + 1) == len(starts_list):
            predicts = predictions[start:]
            Y = labels[start:]
        else:
            predicts = predictions[start:starts_list[i + 1]]
            Y = labels[start:starts_list[i + 1]]

        ave_predicts = np.mean(predicts)
        predicted_index = (ave_predicts // 3.34).astype(np.int8)
        final_decision = labels_pool[predicted_index]
        if Y[0, 0] == 0:
            L = 3.3
        if Y[0, 0] == 10:
            L = 6.6

        if final_decision == Y[0, 0]:
            loss = 0

        else:
            if Y[0,0]==5 and ave_predicts<5:
                loss=(ave_predicts - 3.3) ** 2
            elif Y[0, 0] == 5 and ave_predicts >= 5:
                loss = (ave_predicts - 6.6) ** 2
            else:
                loss = (ave_predicts - L) ** 2

        LOSS = loss + LOSS

    if idx % 15 == 0 or idx==79:
        print('Per Video Regression Error is :%f ' % (LOSS / len(starts_list)))

    return LOSS / len(starts_list)  # returnes the accuracy per video



def vote_accuracy_per_video(labels,predictions,starts_list,idx): #starts_list includes the index of the start index in each video in the main input file
    #VA
    count=0
    labels_pool = np.array([0, 5, 10])
    np.clip(predictions, 0, 10, out=predictions)
    pre=[]
    true=[]
    for i,start in enumerate(starts_list):
        if (i+1)==len(starts_list):
            predicts = predictions[start:]
            Y = labels[start:]
        else:
            predicts=predictions[start:starts_list[i+1]]
            Y=labels[start:starts_list[i+1]]
        true.append(Y[0,0])
        predicted_index = (predicts // 3.34).astype(np.int8)
        predicted_labels = labels_pool[predicted_index]
        alert_voted_percent=np.sum(predicted_labels==labels_pool[0])/len(Y)
        lowVigilant_voted_percent = np.sum(predicted_labels == labels_pool[1]) / len(Y)
        drowsy_voted_percent = np.sum(predicted_labels == labels_pool[2]) / len(Y)
        max=np.max([alert_voted_percent, lowVigilant_voted_percent, drowsy_voted_percent])

        if np.sum([alert_voted_percent, lowVigilant_voted_percent, drowsy_voted_percent]==max)!=1:
            ave_predicts=np.mean(predicts)
            predicted_index = (ave_predicts // 3.34).astype(np.int8)
            final_decision = labels_pool[predicted_index]
        else:
            final_decision=labels_pool[np.argmax([alert_voted_percent,lowVigilant_voted_percent,drowsy_voted_percent])]
        pre.append(final_decision)
        if final_decision==Y[0,0]:
            count=count+1
        if idx%10==0:
            print(str(i+1)+': '+'True label is :%d and the detected label is %d' %(Y[0,0],final_decision))

    f1 = f1_score(true, pre, average='macro')
    precious = precision_score(true, pre, average='macro')
    recall = recall_score(true, pre, average='macro')#计算precious值和recall值，调用了sklearn的函数


    #return count/len(starts_list),f1,precious,recall  #returnes the accuracy per video
    return count/len(starts_list)




def calc_accuracy_per_batch(Y, predicts):  #Y_size=[Batch_size,1]
    labels_pool = np.array([0, 5, 10])
    np.clip(predicts,0,10,out=predicts)
    predicted_index = (predicts // 3.34).astype(np.int8)
    predicted_labels = labels_pool[predicted_index]
    is_correct = np.equal(predicted_labels, Y.astype(np.int8))
    accuracy = np.sum(is_correct)/len(is_correct)

    return accuracy

def batchNorm(x,beta,gamma,training,scope='bn'):
    with tf.variable_scope(scope):
        batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)
        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(training,mean_var_with_update,lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def Network(input,Pre_fc1_size,Post_fc1_size_per_layer,embb_size,embb_size2,Post_fc2_size,hstate_size,num_layers,feature_size,
            step_size,output_size,keep_p,training):
    #input :[Batch,step_size,features]
    #hstate_size=list of hstate_szie for each layer  [layers]

    end_points = {}
    batch_size = tf.shape(input)[0]
    with tf.variable_scope('pre_fc1'):
        pre_fc1_weights=tf.get_variable('weights',[feature_size,Pre_fc1_size],dtype=tf.float32,
                                        initializer=tf.contrib.layers.xavier_initializer(uniform=False,seed=None,dtype=tf.float32))
        pre_fc1_biases = tf.get_variable('biases', [Pre_fc1_size], dtype=tf.float32,
                                          initializer=tf.constant_initializer(0.0))


        reshaped_input_net=tf.reshape(input, [-1, feature_size])
        input_RNN=tf.matmul(reshaped_input_net,pre_fc1_weights)
        input_RNN = batchNorm(input_RNN, pre_fc1_biases, None, training, scope='bn')
        input_RNN=tf.nn.relu(input_RNN)
        input_RNN=tf.reshape(input_RNN,[-1,step_size,Pre_fc1_size]) # size=[batch,Time,Pre_fc1_size ]
        input_RNN=tf.nn.dropout(input_RNN,keep_p)




    end_points['pre_fc1']=input_RNN


    hmslstm_block=Main.HMSLSTM_Block(input_size=[batch_size,step_size,Pre_fc1_size],step_size=step_size,
                                     hstate_size=hstate_size,num_layers=num_layers,keep_p=keep_p)

    output_RNN_set,states_RNN,concati=hmslstm_block(input_RNN,reuse=False)
    end_points['mid_layers'] = output_RNN_set
    with tf.variable_scope('post_fc1'):

        for lay in range(num_layers):
            post_fc1_weights = tf.get_variable('weights_%s' % lay, [hstate_size[lay], Post_fc1_size_per_layer], dtype=tf.float32,
                                               initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=None,
                                                                                                dtype=tf.float32))
            post_fc1_biases = tf.get_variable('biases_%s' % lay, [Post_fc1_size_per_layer], dtype=tf.float32,
                                              initializer=tf.constant_initializer(0.0))
            trash,output_RNN=tf.split(output_RNN_set[lay],[step_size-1,1],axis=0,name='layers')  #size of output_RNN[lay] is (step,batch,hsize),
            #  but we want just the last step
            output_RNN=tf.squeeze(output_RNN,axis=0) #size=(batch,h_size)
            post_fc1 = tf.matmul(output_RNN, post_fc1_weights)
            post_fc1 = batchNorm(post_fc1, post_fc1_biases,None, training, scope='bn')

            if lay==0:
                post_fc1_out=post_fc1
            else:
                post_fc1_out=tf.concat([post_fc1_out,post_fc1],axis=1) #size=[Batch,layer*Post_fc1_size_per_layer]

        post_fc1_out=tf.nn.relu(post_fc1_out)
        post_fc1_out = tf.nn.dropout(post_fc1_out,keep_p)
        end_points['post_fc1'] = post_fc1_out
    with tf.variable_scope('embeddings'):
        embeddings_weights = tf.get_variable('weights' , [Post_fc1_size_per_layer*num_layers,embb_size], dtype=tf.float32,
                                             initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=None,
                                                                                              dtype=tf.float32))
        embeddings_biases = tf.get_variable('biases' , [embb_size], dtype=tf.float32,
                                          initializer=tf.constant_initializer(0.0))



        embeddings = tf.matmul(post_fc1_out, embeddings_weights)
        embeddings = batchNorm(embeddings, embeddings_biases, None, training, scope='bn')
        embeddings = tf.nn.relu(embeddings)
        embeddings = tf.nn.dropout(embeddings, keep_p)
        end_points['embeddings'] = embeddings
    with tf.variable_scope('embeddings2'):
        embeddings_weights2 = tf.get_variable('weights' , [embb_size,embb_size2], dtype=tf.float32,
                                             initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=None,
                                                                                              dtype=tf.float32))
        embeddings_biases2 = tf.get_variable('biases' , [embb_size2], dtype=tf.float32,
                                          initializer=tf.constant_initializer(0.0))



        embeddings2 = tf.matmul(embeddings, embeddings_weights2)
        embeddings2 = batchNorm(embeddings2, embeddings_biases2, None, training, scope='bn')
        embeddings2 = tf.nn.relu(embeddings2)
        embeddings2 = tf.nn.dropout(embeddings2, keep_p)
        end_points['embeddings2'] = embeddings2

    with tf.variable_scope('post_fc2'):
        post_fc2_weights = tf.get_variable('weights' , [embb_size2, Post_fc2_size],
                                                 dtype=tf.float32,
                                           initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=None,
                                                                                            dtype=tf.float32))
        post_fc2_biases = tf.get_variable('biases', [Post_fc2_size], dtype=tf.float32,
                                                initializer=tf.constant_initializer(0.0))

        post_fc2_out = tf.matmul(embeddings2 , post_fc2_weights)
        post_fc2_out = batchNorm(post_fc2_out,post_fc2_biases, None, training, scope='bn')
        post_fc2_out=tf.nn.relu(post_fc2_out)
        post_fc2_out = tf.nn.dropout(post_fc2_out, keep_p)
        end_points['post_fc2'] = post_fc2_out
    with tf.variable_scope('last_fc'):
        last_fc_weights = tf.get_variable('weights' , [Post_fc2_size,output_size],
                                           dtype=tf.float32,
                                          initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=None,
                                                                                           dtype=tf.float32))
        last_fc_biases = tf.get_variable('biases', [output_size], dtype=tf.float32,
                                          initializer=tf.constant_initializer(0.0))

        output = tf.matmul(post_fc2_out, last_fc_weights)+last_fc_biases


        if output_size==1:
            end_points['before the last sigmoid'] = output
            output = 10 * tf.sigmoid(output)


    return output,end_points,concati #size=[Batch,1]



########
########

def batch_gen(data,label,batch_size):  # data=[Total data points, T,F]   #label=[Total Data Points, 1]
    n=len(data)
    batch_num=n // batch_size
    for b in range(batch_num):  # Here it generates batches of data within 1 epoch consecutively
        X=data[batch_size*b:batch_size*(b+1),:,:]
        Y= label[batch_size * b:batch_size * (b + 1),:]
        yield X,Y
    if n> batch_size * (b + 1):
        X = data[batch_size * (b + 1):, :, :]
        Y = label[batch_size * (b + 1):, :]
        yield X, Y



def epoch_gen(data,label,batch_size,num_epochs): # data=[Total data points, T,F]  # This generates epochs of batches of data
    for epoch in range(num_epochs): # Inside one epoch
        yield batch_gen(data,label,batch_size)

def save_variables(sess,path,f):
        saver = tf.train.Saver()
        print('saving variables...\n')
        saver.save(sess,path+'my_model%d'%f)


def save_variablesbest(sess, path,f):
    saver = tf.train.Saver()
    print('saving variables...\n')
    saver.save(sess, path + 'my_model_best_%d'%f)

def Train(total_input,total_labels,TestB,TestL,output_size,feature_size,batch_size,num_epochs,Pre_fc1_size,Post_fc1_size_per_layer,embb_size,
          embb_size2,Post_fc2_size,hstate_size,num_layers,step_size,drop_out_p,lr,th,start_i,load,fold_num):  #total_input is the shuffled input with size=[Total data points, T,F]

    #shape of total_input [N,T,F]
    #shape of total_labels=[N,1]
    #TestB:the test blink sequences
    #TestL: the test labels
    #feature_size:input feature_size
    #Order of layers==>  Pre_fc1---HMLSTM---Post_fc1----embb----embb2----Post_fc2---out
    #step_size:the time steps of the HMLSTM network
    #lr:leanining rate
    #th: the Delta for the cost function
    #start_i: the start indices of blink sequences in each test video
    #load: if True loads the weights from disk[Binary]
    #fold_num: decides that the model used is the one trained on all the folds except fold_num (if load==True)
            #  decides that the test model (if load==False)
    tf.reset_default_graph()
    L2loss=0
    input_net = tf.placeholder(tf.float32, shape=(None, None, feature_size), name='bacth_in')
    labels = tf.placeholder(tf.float32, shape=(None, output_size), name='labels_net')  #size=[batch,1]
    keep_p=tf.placeholder(tf.float32)
    training = tf.placeholder(tf.bool,name='phase_train')
    output,end_points,concati=Network(input=input_net,Pre_fc1_size=Pre_fc1_size,Post_fc1_size_per_layer=Post_fc1_size_per_layer,
                   embb_size=embb_size,embb_size2=embb_size2,Post_fc2_size=Post_fc2_size,hstate_size=hstate_size,num_layers=num_layers,
                   feature_size=feature_size,step_size=step_size,output_size=output_size,keep_p=keep_p,training=training)
    error=tf.abs(output-labels)
    loss2 =tf.maximum(0.0,tf.square(error)-th)
    loss2 = tf.reduce_mean(loss2)
    variable_path= 'xbearw50o30wei1model/'
    with tf.variable_scope('last_fc',reuse=True):
        last_fc_weights = tf.get_variable('weights')
    with tf.variable_scope('post_fc2',reuse=True):
        post_fc2_weights = tf.get_variable('weights')
    with tf.variable_scope('embeddings',reuse=True):
        embeddings_weights = tf.get_variable('weights')
    with tf.variable_scope('embeddings2',reuse=True):
        embeddings_weights2 = tf.get_variable('weights')
    with tf.variable_scope('pre_fc1',reuse=True):
        pre_fc1_weights = tf.get_variable('weights')

    with tf.variable_scope('post_fc1',reuse=True):
        for lay in range(num_layers):
            post_fc1_weights = tf.get_variable('weights_%s' % lay)
            L2loss=tf.nn.l2_loss(post_fc1_weights)+L2loss
    #
    loss=loss2+0.1 * (tf.nn.l2_loss(last_fc_weights) +tf.nn.l2_loss(pre_fc1_weights) + L2loss+
                       tf.nn.l2_loss(post_fc2_weights) + tf.nn.l2_loss(embeddings_weights)+ tf.nn.l2_loss(embeddings_weights2))
    optimizer=tf.train.AdamOptimizer(lr).minimize(loss)
    with tf.Session() as sess:
        ###此处需注意加载的名称
        if(load==True):
            saver = tf.train.Saver()
            print('loading variables...')
            saver.restore(sess, variable_path+'my_model_best_%d'%fold_num)
            #saver.restore(sess, variable_path + 'my_model%d' % fold_num)
        else:
            sess.run(tf.global_variables_initializer())
        ####plotting_setup\
        y1 = np.zeros([num_epochs])
        yy = np.zeros([num_epochs])
        y_test = np.zeros([num_epochs])
        yy_test = np.zeros([num_epochs])
        y_v = np.zeros([num_epochs])
        y_best=np.zeros([num_epochs])#用来存放最好的那一次va的epoch次数
        max=0
        ####
        for idx,epoch in enumerate(epoch_gen(data=total_input,label=total_labels,batch_size=batch_size,num_epochs=num_epochs)):#set of batches in one epoch
            loss_per_epoch = 0
            sum=0
            if load==False:
                for b,(X,Y) in enumerate(epoch):# each batch in each epoch

                        if output_size==1:
                            _,loss_values,predicts,mid_values,concat=sess.run([optimizer,loss,output,end_points,concati],
                                                                           feed_dict={input_net: X, labels: Y,keep_p:drop_out_p,training:True})
                            accuracy = calc_accuracy_per_batch(Y, predicts) #BSA

                        loss_per_epoch= loss_values +loss_per_epoch
                        sum=sum+accuracy                                # calculating the moving average for acc
                        moving_accuracy=sum/(b+1)
                        if b%50==0:
                            print('Epoch number: ' + str(idx) + ' ---- ' + 'Batch number ' + str(
                                b) + ' -- the loss is: ' + str(loss_values) + '--the accuracy: '+str(moving_accuracy)+'\n')
            ###########TEsting
            if output_size==1:
                loss_values_Test, predicts_Test,mid_vT = sess.run([loss, output,end_points],feed_dict={input_net: TestB, labels: TestL,keep_p:1.0,training:False})
                accuracy_Test = calc_accuracy_per_batch(TestL, predicts_Test) #BSA
                #accuracy_per_videoV=vote_accuracy_per_video(TestL,predicts_Test,start_i,idx) #VA
                # accuracy_per_videoV, f1, precious, recall = vote_accuracy_per_video(TestL, predicts_Test, start_i,
                #                                                                      idx)  # VA
                accuracy_per_videoV= vote_accuracy_per_video(TestL, predicts_Test, start_i,idx)
                ###########
            ###########

            if load==False:
                print("For Training: "+str(loss_per_epoch)+" , "+str(moving_accuracy))
                yy[idx] = moving_accuracy
                y1[idx] = loss_per_epoch
            print("BSA: " + str(loss_values_Test) + " , " + str(accuracy_Test))

            print("VA: " + str(accuracy_per_videoV))
            # print("F1: " + str(f1))
            # print("precious: " + str(precious))
            # print("recall: " + str(recall))
            regression_per_video(TestL, predicts_Test, start_i, idx) #VRE
            print("----------------------------------------------")


            yy_test[idx]=accuracy_Test
            y_test[idx]=loss_values_Test
            y_v[idx]=accuracy_per_videoV
            if load==False:
                if idx == 0:
                    max = 0
                    save_variablesbest(sess, variable_path, fold_num)
                if y_v[idx] > y_v[max]:
                    max = idx
                    save_variablesbest(sess, variable_path, fold_num)
                y_best[idx] = max
            #save_variablesbest(sess,variable_path,fold_num)
            '''
            x = np.linspace(0, num_epochs, num_epochs)
            y = yy_test
            yb = y_v
            # 在生成的坐标系下画折线图
            fig, ax = plt.subplots()
            line1, = ax.plot(x, y, 'm', label="BSA", linestyle='--')
            line2, = ax.plot(x, yb, 'b', label="VA", linewidth=1)

            # Create a legend for the first line.
            first_legend = ax.legend(handles=[line1], loc='upper right')

            # Add the legend manually to the Axes.
            ax.add_artist(first_legend)

            # Create another legend for the second line.
            ax.legend(handles=[line2], loc='lower right')

            plt.show()
            '''
            if load==True:
                break
            if idx % 5 == 0:
                print("Sav")
        if load==False:
            save_variables(sess,variable_path,fold_num) # Saving every epoch



    return y1,yy,y_test,yy_test,y_v,y_best





################################################
load=False  #Load the weights from disk if True
for i in range(5): #Cross validation but recommended to run each fold a few times to see the best perfomrance as you may
    # get caught up in a local minimum

    ii=i    # ii decides the model and i decides the fold_num for test
    if load==True:
        ii=i
    Blinks = np.load('xbearw50o30wei1npy/xiaobo_30_Fold%d.npy'%(i+1))
    Labels = np.load('xbearw50o30wei1npy/Labels_30_Fold%d.npy'%(i+1))
    BlinksTest = np.load('xbearw50o30wei1npy/xiaoboTest_30_Fold%d.npy'%(i+1))
    LabelsTest = np.load('xbearw50o30wei1npy/LabelsTest_30_Fold%d.npy'%(i+1))
    #deciding the indices of each video based on the fold
    #####################Normalizing the input#############Second phase
    BlinksTest[:,:,0]=(BlinksTest[:,:,0]-np.mean(Blinks[:,:,0]))/np.std(Blinks[:,:,0])
    Blinks[:,:,0]=(Blinks[:,:,0]-np.mean(Blinks[:,:,0]))/np.std(Blinks[:,:,0])
    #####
    #####
    BlinksTest[:,:,1]=(BlinksTest[:,:,1]-np.mean(Blinks[:,:,1]))/np.std(Blinks[:,:,1])
    Blinks[:,:,1]=(Blinks[:,:,1]-np.mean(Blinks[:,:,1]))/np.std(Blinks[:,:,1])

    if i==0:
        start_indices=[0, 187, 472, 758, 947, 1234, 1522, 1739, 2033, 2327, 2484, 2772, 3014, 3205, 3492, 3779, 3934, 4138, 4294, 4480, 4764, 5049, 5237, 5523, 5808, 5995, 6287, 6603, 6770, 7066, 7366, 7561, 7710, 7873, 8047, 8347]

    if i==1:
        start_indices=[0, 188, 515, 809, 999, 1314, 1611, 1811, 2106, 2400, 2512, 2739, 2966, 3153, 3443, 3732, 3848, 3983, 4097, 4283, 4569, 4856, 5058, 5345, 5688, 5876, 6162, 6449, 6637, 6927, 7218, 7399, 7670, 7955, 8064, 8232]

    if i==2:
        start_indices=[0, 186, 470, 755, 946, 1372, 1704, 1909, 2207, 2503, 2684, 2912, 3147, 3559, 3681, 3845, 4058, 4391, 4681, 4856, 5094, 5348, 5454, 5669, 5926, 6108, 6244, 6507, 6622, 6953, 7245, 7437, 7727, 8044, 8238, 8537]


    if i==3:
        start_indices=[0, 214, 509, 835, 1022, 1309, 1596, 1797, 2085, 2373, 2560, 2846, 3132, 3327, 3492, 3950, 4136, 4419, 4670, 4857, 5145, 5430, 5498, 5606, 5713, 5785, 5921, 6061, 6243, 6529, 6811, 6997, 7283, 7574, 7770, 8059]


    if i==4:
        start_indices=[0, 188, 535, 854, 1042, 1334, 1628, 1814, 2101, 2388, 2575, 2873, 3171, 3369, 3536, 3837, 4058, 4364, 4624, 4814, 5106, 5405, 5601, 5890, 6182, 6368, 6659, 6948, 7135, 7423, 7710, 7897, 8175]



    print('######################')
    print(i)
    print('######################')
    start_indices=np.asarray(start_indices)
    loss,accuracy,loss_Test,accuracy_Test,acc_per_Vid,best_epoch=Train(total_input=Blinks,total_labels=Labels,TestB=BlinksTest,TestL=LabelsTest,
                    output_size=1,feature_size=2,batch_size=64,num_epochs=200,Pre_fc1_size=32,Post_fc1_size_per_layer=16,
                    embb_size=16,embb_size2=16,Post_fc2_size=8,hstate_size=[32,32,32,32,32,32],num_layers=6,step_size=30,drop_out_p=1.0,
                                                  lr=0.0003,th=1.253,start_i=start_indices,load=load,fold_num=ii)

    #
    # if load==False:
    #     #np.save(open('mfcc5modelw100o60best/x%d.npy' %ii, 'wb'),x)
    #     np.save(open('xbearw50o30wei1model/loss%d.npy'%ii, 'wb'),loss) #for training
    #     np.save(open('xbearw50o30wei1model/accuracy%d.npy' %ii, 'wb'),accuracy) #for training
    #     np.save(open('xbearw50o30wei1model/loss%dTest.npy'%ii, 'wb'),loss_Test) #for test
    #     np.save(open('xbearw50o30wei1model/accuracy%dTest.npy'%ii, 'wb'),accuracy_Test) #for test (BSA)
    #     np.save(open('xbearw50o30wei1model/accuracy%dVTest.npy'%ii, 'wb'),acc_per_Vid) #for test    (VA)
    #     np.save(open('xbearw50o30wei1model/bestepoch%d.npy'%ii, 'wb'),best_epoch)

