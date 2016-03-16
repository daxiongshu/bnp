import pandas as pd
import numpy as np
import tensorflow as tf

import scipy as sp
def logloss(act, pred):
    epsilon = 1e-6
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    #print np.mean(pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from six.moves import cPickle as pickle

BETA=1e-5
LEARNING_RATE = 1e-2
BATCH_SIZE = 256
NUM_FEATURE = 131
NUM_LABEL =2
train_path = '../cv/train_clean1.csv'
test_path = '../cv/test_clean1.csv'
def dense_to_one_hot(labels_dense, num_classes=10):
  """
  Convert class labels from scalars to one-hot vectors.
  http://stackoverflow.com/questions/33681517/tensorflow-one-hot-encoder
  """
  labels_dense = np.array(labels_dense)
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def onehot(y):
    tmp=np.unique(y)
    yp=np.zeros((y.shape[0],tmp.shape[0]))
    for c,i in enumerate(tmp):
        yp[y==i,c]=1
    return yp
def process_data2(path, is_test_set=False,split=True):
    df = pd.read_csv(path)
    if not is_test_set:
        labels = df['target'].values
        labels = onehot(labels)
        data=df.drop(['ID','target'], axis=1).as_matrix()
        sc=StandardScaler()
        data=sc.fit_transform(data)
        pickle.dump(sc,open('sc.p','w'))
        X_train, X_test, y_train, y_test = train_test_split(data, labels,
                                                test_size=0.15, random_state=42)
        if split:
            return X_train, X_test, y_train, y_test
        else:
            return data,labels
    else:
        ids = df['ID'].values#.tolist()
        data = df.drop('ID', axis=1).as_matrix()
        sc=pickle.load(open('sc.p'))
        data=sc.transform(data)
        return data, ids
def forward_nn(X,y):
    # input is tensor
    num=10 # number of hidden layer1 neurons
    num2=5 # number of hidden layer2 neurons
    weights = tf.Variable(tf.truncated_normal([int(X.get_shape()[1]), num]))
    biases = tf.Variable(tf.zeros([num]))
    
    d1=tf.matmul(X,weights)+biases
    d2=tf.nn.relu(d1)
    
    weights1 = tf.Variable(tf.truncated_normal([num,num2]))
    biases1 = tf.Variable(tf.zeros([num2]))
    
    d1=tf.matmul(d2,weights1)+biases1
    d2=tf.nn.relu(d1)
    
    weights2 = tf.Variable(tf.truncated_normal([num2,int(y.get_shape()[1])]))
    biases2 = tf.Variable(tf.zeros([y.get_shape()[1]]))
    logit=tf.matmul(d2,weights2)+biases2
    
    l2loss=[tf.nn.l2_loss(i) for i in [weights,biases,weights2,biases2]]
    l2loss =tf.add_n(l2loss) #tf.reshape(tf.concat(1, l2loss), [-1, 4])
    #l2loss=tf.reduce_sum(l2loss)
    return tf.nn.softmax(logit),logit,l2loss
def next_batch(X,y,start,batch_size=128):
    start=start%X.shape[0]
    end=min(start+batch_size,X.shape[0])
    return X[start:end],y[start:end]
def tensor_train_predict(X,y,Xt,yt=None):
    # input is numpy array
    
    X_ =  tf.placeholder(tf.float32,shape=[None,X.shape[1]])
    y_ =  tf.placeholder(tf.float32,shape=[None,y.shape[1]])
    
    yp,logit,l2loss=forward_nn(X_,y_)
    #cross_entropy = -tf.reduce_mean(y_*tf.log(yp)) # this lost precision since float32 is used
    # always use logit! more precise!
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logit, y_)) #+ l2loss*BETA
    #train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
    
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    
    train_loss=[]
    val_loss=[]
    for i in range(50000):
        batch_xs, batch_ys = next_batch(X,y,(i)*128,batch_size=128)
        _,ytmp=sess.run([train_step,yp], feed_dict={X_: batch_xs, y_: batch_ys})
        
        if i%1000==0:
            train_loss.append(cross_entropy.eval(session=sess,feed_dict={X_: batch_xs, y_: batch_ys}))
            if yt is not None:
                val_loss.append(cross_entropy.eval(session=sess,feed_dict={X_: Xt, y_: yt}))
            print i,np.mean(ytmp),np.mean(batch_ys),'train loss:',train_loss[-1],
            if yt is not None:
                print 'validation loss:',val_loss[-1]
            else:
                print
    if yt is None:   
        yt=np.zeros([Xt.shape[0],2])
    result=sess.run([yp],feed_dict={X_: Xt, y_: yt})
    sess.close()
    return result

X,y = process_data2(train_path,split=False)
Xt,ids= process_data2(test_path,is_test_set=True)   
print X.shape, y.shape, Xt.shape, ids.shape
yp=tensor_train_predict(X,y,Xt)[0][:,1]
s=pd.DataFrame({'ID':ids,'PredictedProb':yp})
s.to_csv('tensor1.csv',index=False)
 
