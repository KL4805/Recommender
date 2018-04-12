
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import heapq
import time
from math import *


# In[2]:


train_data = []
with open('train_data.txt','r') as infile:
    for l in infile:
        l.rstrip('\n')
        lint = [int(x) for x in l.split(',')]
        train_data.append(lint)


# In[3]:


item_of_user = [[] for i in range(6040)]
for l in train_data:
    u = l[0]
    item = l[1]
    item_of_user[u].append(item)


# In[4]:


test_data = []
with open('test_data.txt','r') as infile:
    for l in infile:
        l.rstrip('\n')
        lint = [int(x) for x in l.split(',')]
        test_data.append(lint)


# In[5]:


negative_pair = [[] for i in range(6040)]
with open('negative_pairs.txt','r') as infile:
    for l in infile:
        l.rstrip('\n')
        u,i = (int(x) for x in l.split(','))
        negative_pair[u].append(i)


# In[6]:


num_user = 6040
num_item = 3952


# In[7]:


#For this task, we are performing a regression-like task, therefore the batch contains users, items and labels. 
def get_data(negative_num):
    batch_user, batch_item, batch_label = [], [], []
    for u in range(num_user):
        for it in item_of_user[u]:
            batch_user.append(u)
            batch_item.append(it)
            batch_label.append(1)
            sampled_negative = 0
            while sampled_negative <negative_num:
                j = np.random.randint(num_item)
                if j in item_of_user[u]:
                    continue
                else:
                    batch_user.append(u)
                    batch_item.append(j)
                    batch_label.append(0)
                    sampled_negative += 1
    
    return np.array(batch_user), np.array(batch_item), np.array(batch_label)        


# In[8]:


#The first layer is an embedding layer, from then on are fully connected layers. 
latent_size = 32
batch_size = 256
learning_rate = 0.01
regularization_rate = 1e-20
max_epochs = 20
negative_ratio = 4
layer_size = [64, 32, 16, 8]
#latent_size*2 = layer_size[0]
#the layer_size does not include the last output layer


# In[9]:


tf.reset_default_graph()
embedding_u = tf.get_variable(name = 'MLP_latent_u', shape = [num_user, latent_size], initializer = tf.random_normal_initializer(stddev = 0.01))
embedding_i = tf.get_variable(name = 'MLP_latent_i', shape = [num_item, latent_size], initializer = tf.random_normal_initializer(stddev = 0.01))
#w1 = tf.get_variable(name = 'fc-w1', shape = [layer_size[0], layer_size[1]], initializer = tf.random_normal_initializer(stddev = 0.01))
#b1 = tf.get_variable(name = 'fc-b1', shape = [layer_size[1]], initializer = tf.constant_initializer(0.001))
#w2 = tf.get_variable(name = 'fc-w2', shape = [layer_size[1], layer_size[2]], initializer = tf.random_normal_initializer(stddev = 0.01))

hidden_weights = []
hidden_biases = []
for i in range(len(layer_size) - 1):
    hidden_weights.append(tf.get_variable(name = "MLP_hidden_w"+str(i), shape = [layer_size[i], layer_size[i+1]], initializer = tf.random_normal_initializer))
    hidden_biases.append(tf.get_variable(name = 'MLP_hidden_b'+str(i), shape = [layer_size[i+1]], initializer = tf.constant_initializer(0.01)))
output_weight = tf.get_variable(name = 'MLP_output_w', shape = [layer_size[-1], 1], initializer = tf.random_normal_initializer(stddev = 0.01))


# In[10]:


U = tf.placeholder(tf.int32, [None])
I = tf.placeholder(tf.int32, [None])
label = tf.placeholder(tf.int32, [None])

regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)

embedding_u = tf.nn.embedding_lookup(embedding_u, U)
embedding_i = tf.nn.embedding_lookup(embedding_i, I)
concat_ui = tf.concat([embedding_u, embedding_i], axis = 1)

#hidden_out = tf.nn.relu(tf.matmul(concat_ui, w1) + b1)
hidden_in = concat_ui
for i in range(len(layer_size) - 1):
    hidden_out = tf.nn.relu(tf.matmul(hidden_in, hidden_weights[i]) + hidden_biases[i])
    hidden_in = hidden_out
    tf.add_to_collection("losses", regularizer(hidden_weights[i]))

score = tf.reshape(tf.matmul(hidden_out, output_weight), [-1])
prediction = tf.nn.sigmoid(score)
reg_term = regularizer(embedding_u) + regularizer(embedding_i) + tf.add_n(tf.get_collection("losses"))
loss = reg_term + tf.losses.log_loss(predictions = prediction, labels = label)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


# In[11]:


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    best_hr = 0
    for ep in range(max_epochs):
        t1 = time.time()
        eu,ei,el = get_data(negative_ratio)
        for j in range(len(eu)//batch_size):
            sample = np.random.randint(len(eu), size = batch_size)
            #print(sample)
            bu = eu[sample]
            bi = ei[sample]
            bl = el[sample]
            #print(bu)
            #print(bi)
            #print(bl)
            loss_value ,_ = sess.run([loss, optimizer], feed_dict = {U:bu, I:bi, label:bl}) 
            #print(grad_value)
    
        t2 = time.time()
        print("[%.1f s] After %d epochs, loss on batch is %.3f."%(t2-t1, ep, loss_value))
        
        hits = 0
        tNDCG = 0
        for u in range(num_user):
            map_item_rating = {}
            maxScore = sess.run(score, feed_dict = {U:np.array([u]), I:np.array([test_data[u][1]])})
            negative_i = negative_pair[u]
            negative_u = [u for m in range(100)]
            negative_score = sess.run(score, feed_dict = {U:negative_u, I:negative_i})
            map_item_rating[test_data[u][1]] = maxScore
            for k in range(100):
                map_item_rating[negative_i[k]] = negative_score[k]
            ranklist = heapq.nlargest(10, map_item_rating, key = map_item_rating.get)
            if test_data[u][1] in ranklist:
                hits +=1
                idx = ranklist.index(test_data[u][1])
                tNDCG += log(2)/log(idx+2)
        
        hr = hits/6040
        NDCG = tNDCG/6040
        print("[%.1f s] After %d epochs, hit@10 = %.3f, NDCG@10 = %.3f."%(time.time()-t1, ep, hits/6040, tNDCG/6040))
       
        if hr > best_hr:
            best_hr = hr
            saver = tf.train.Saver()
            saver.save(sess, "MLPModel/model.ckpt", global_step = ep)

