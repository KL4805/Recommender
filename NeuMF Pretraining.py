
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import heapq
from time import time
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


# In[15]:


GMF_latent_size = 32
MLP_latent_size = 32
learning_rate = 0.0005
max_epochs = 20
MLP_layers = [64, 32, 16, 8]
#MLP_layers[0] = 2*MLP_latent_size
regularization_rate = 1e-20
batch_size = 256
negative_ratio = 4

alpha = 0.5
# the tradeoff between GMF and MLP final output weights


# In[16]:


tf.reset_default_graph()
GMF_embedding_u = tf.get_variable(name = "GMF_latent_u", shape = [num_user, GMF_latent_size])
GMF_embedding_i = tf.get_variable(name = "GMF_latent_i", shape = [num_item, GMF_latent_size])
GMF_weight = tf.get_variable(name = "GMF_weight", shape = [GMF_latent_size, 1])


MLP_embedding_u = tf.get_variable(name = "MLP_latent_u", shape = [num_user, MLP_latent_size])
MLP_embedding_i = tf.get_variable(name = "MLP_latent_i", shape = [num_item, MLP_latent_size])

MLP_hidden_weights = []
MLP_hidden_biases = []
for i in range(len(MLP_layers) - 1):
    MLP_hidden_weights.append(tf.get_variable(name = 'MLP_hidden_w' + str(i), shape = [MLP_layers[i], MLP_layers[i+1]]))
    MLP_hidden_biases.append(tf.get_variable(name = "MLP_hidden_b" + str(i), shape = [MLP_layers[i+1]]))

MLP_output = tf.get_variable(name = 'MLP_output_w', shape = [MLP_layers[-1], 1])


# In[17]:


U = tf.placeholder(tf.int32, [None])
I = tf.placeholder(tf.int32, [None])
label = tf.placeholder(tf.int32, [None])
regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)


# In[18]:


#GMF_part
GMF_embedded_u = tf.nn.embedding_lookup(GMF_embedding_u, U)
GMF_embedded_i = tf.nn.embedding_lookup(GMF_embedding_i, I)

GMF_merged_ui = tf.multiply(GMF_embedded_u, GMF_embedded_i)


# In[19]:


#MLP_part
MLP_embedded_u = tf.nn.embedding_lookup(MLP_embedding_u, U)
MLP_embedded_i = tf.nn.embedding_lookup(MLP_embedding_i, I)

MLP_merged_ui = tf.concat([MLP_embedded_u, MLP_embedded_i], axis = 1)

hidden_in = MLP_merged_ui

for i in range(len(MLP_layers) - 1):
    hidden_out = tf.nn.relu(tf.matmul(hidden_in, MLP_hidden_weights[i]) + MLP_hidden_biases[i])
    hidden_in = hidden_out
    tf.add_to_collection("losses", regularizer(MLP_hidden_weights[i]))


# In[20]:


final_concat = tf.concat([GMF_merged_ui, hidden_out], axis = 1)
total_weight = tf.concat([GMF_weight*alpha, (1-(alpha))*MLP_output], axis = 0)
output = tf.reshape(tf.matmul(final_concat, total_weight), [-1])
prediction = tf.nn.sigmoid(output)
reg_term = regularizer(GMF_embedding_u) +regularizer(GMF_embedding_i) + regularizer(MLP_embedding_u) + regularizer(MLP_embedding_i)
reg_term += tf.add_n(tf.get_collection("losses"))
loss = reg_term + tf.losses.log_loss(labels = label, predictions = prediction)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


# In[21]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #load the pretrained weights
    best_hr = 0
    saver1 = tf.train.Saver([GMF_embedding_u, GMF_embedding_i, GMF_weight])
    saver2 = tf.train.Saver([MLP_embedding_u, MLP_embedding_i] + MLP_hidden_weights + MLP_hidden_biases + [MLP_output])
    saver1.restore(sess, "GMFModel/model.ckpt-14")
    saver2.restore(sess, "MLPModel/model.ckpt-16")
    #print(GMF_embedding_u.eval())
    for ep in range(max_epochs):
        t1 = time()
        eu, ei, el = get_data(negative_ratio)
        for j in range(len(eu)//batch_size):
            sample = np.random.randint(len(eu), size = batch_size)
            bu = eu[sample]
            bi = ei[sample]
            bl = el[sample]
            loss_value, _ = sess.run([loss, optimizer], feed_dict = {U:bu, I:bi, label:bl})
        
        t2 = time()
        print("[%.1f s] After %d epochs, loss on batch is %.3f."%(t2-t1, ep, loss_value))
        
        hits = 0
        tNDCG = 0
        for u in range(num_user):
            map_item_rating = {}
            maxScore = sess.run(output, feed_dict = {U:np.array([u]), I:np.array([test_data[u][1]])})
            negative_i = negative_pair[u]
            negative_u = [u for m in range(100)]
            negative_score = sess.run(output, feed_dict = {U:negative_u, I:negative_i})
            map_item_rating[test_data[u][1]] = maxScore
            for k in range(100):
                map_item_rating[negative_i[k]] = negative_score[k]
            ranklist = heapq.nlargest(10, map_item_rating, key = map_item_rating.get)
            if test_data[u][1] in ranklist:
                hits +=1
                idx = ranklist.index(test_data[u][1])
                tNDCG += log(2)/log(idx+2)
        
        print("[%.1f s] After %d epochs, hit@10 = %.3f, NDCG@10 = %.3f."%(time()-t1, ep, hits/6040, tNDCG/6040))

