
# coding: utf-8

# In[ ]:


#Bayesian Personalized Ranking, provided at Xiangnan He, Neural Collaborative Filtering as a baseline


# In[ ]:


#For Reference: 
#Factor = 8, HitRate = 0.63, NDCG = 0.36
#Factor = 16, HR = 0.66, NDCG = 0.39
#Factor = 32, HR = 0.68, NDCG = 0.41
#Factor = 64, HR = 0.68, NDCG = 0.41-0.42


# In[1]:


import tensorflow as tf
import numpy as np


# In[2]:


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# In[3]:


train_data = []
with open('train_data.txt', 'r') as infile:
    for l in infile:
        l.rstrip('\n')
        intl = [int(x) for x in l.split(',')]
        train_data.append(intl)


# In[4]:


num_user = 6040
num_item = np.max([u[1] for u in train_data])+1


# In[5]:


num_records = len(train_data)


# In[6]:


test_data = [0 for i in range(6040)]
with open('test_data.txt','r') as infile:
    for l in infile:
        l.rstrip('\n')
        intl = [int(x) for x in l.split(',')]
        test_data[intl[0]] = intl[1]


# In[7]:


negative_pair = [[] for i in range(6040)]
with open('negative_pairs.txt','r') as infile:
    for l in infile:
        l.rstrip('\n')
        u,i = (int(x) for x in l.split(','))
        negative_pair[u].append(i)


# In[8]:


#preprocessing the data
item_of_user = [[] for i in range(6040)]
for d in train_data:
    item_of_user[d[0]].append(d[1])


# In[15]:


def get_batch(batch_size):
    u_batch, i_batch, j_batch = [], [], []
    for i in range(batch_size):
        u = np.random.randint(0, num_user)
        i = np.random.randint(0, len(item_of_user[u]))
        pi = item_of_user[u][i]
        j = np.random.randint(0, num_item)
        while j in item_of_user[u]:
            j = np.random.randint(0, num_item)
        u_batch.append(u)
        i_batch.append(pi)
        j_batch.append(j)
    return u_batch, i_batch, j_batch


# In[19]:


num_epochs = 50
batch_size = 320
learning_rate = 0.01
regularization_rate = 0.01
latent_size = 8


# In[20]:


tf.reset_default_graph()
gamma_u = tf.get_variable(name = "latent-u", shape = [num_user, latent_size], initializer = tf.truncated_normal_initializer(stddev = 0.01))
gamma_i = tf.get_variable(name = 'latent-i', shape = [num_item, latent_size], initializer = tf.truncated_normal_initializer(stddev = 0.01))
#gamma_u = tf.get_variable(name = 'latent_u', shape = [num_user, latent_size], initializer = tf.constant_initializer(0.5))
#gamma_i = tf.get_variable(name = 'latent_i', shape = [num_user, latent_size], initializer = tf.constant_initializer(0.5))


# In[21]:


regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)
U = tf.placeholder(tf.int32, [None])
I = tf.placeholder(tf.int32, [None])
J = tf.placeholder(tf.int32, [None])
latent_batch_u = tf.nn.embedding_lookup(gamma_u, U)
latent_batch_i = tf.nn.embedding_lookup(gamma_i, I)
latent_batch_j = tf.nn.embedding_lookup(gamma_i, J)
x_uij = tf.reduce_sum(tf.multiply(latent_batch_u, latent_batch_i-latent_batch_j), 1)
#neg_mark = tf.einsum('ij,ij->i', latent_batch_u, latent_batch_j)
#x_uij = tf.nn.sigmoid(pos_mark-neg_mark)
regu = regularizer(latent_batch_u) + regularizer(latent_batch_i)+regularizer(latent_batch_j)
loss = regu - tf.reduce_sum(tf.log(tf.nn.sigmoid(x_uij)))
train_optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
#evaluation_score = tf.reduce_sum(tf.multiply(latent_batch_u, latent_batch_j), 1)


# In[22]:


from math import *
from time import *
import heapq


# In[23]:


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(num_epochs):
        t1 = time()
        for j in range(num_records//batch_size):
            batch_u,batch_i, batch_j = get_batch(batch_size)
            _, reg_term, loss_value = sess.run([train_optimizer, regu, loss], feed_dict = {U:batch_u, I:batch_i, J:batch_j})
            #print(xuij)
        print("[%.1f s] After %d epochs, loss is %f while regularize term is %f"%(time()-t1, i, loss_value, reg_term))
        #test it using the negative pairs
        gamma_u_val = gamma_u.eval()
        gamma_i_val = gamma_i.eval()
        #if i==0: 
        #    print(gamma_u_val[0])
        #if i==10:
        #    print(gamma_u_val[0])
        
        
        hits = 0
        tNDCG = 0 
        for k in range(num_user):
            map_item_rating = {}
            maxScore = np.dot(gamma_u_val[k], gamma_i_val[test_data[k]])
            map_item_rating[test_data[k]] = maxScore
            early_stop = False
            countLarger = 0
            for m in range(100):
                _score = np.dot(gamma_u_val[k], gamma_i_val[negative_pair[k][m]])
                map_item_rating[negative_pair[k][m]] = _score
                
                if _score>maxScore:
                    countLarger +=1
                
                if countLarger >=10:
                    early_stop = True
                    break
            if early_stop == False:
                ranklist = heapq.nlargest(10, map_item_rating, key = map_item_rating.get)
                if test_data[k] in ranklist:
                    hits += 1
                    idx = ranklist.index(test_data[k])
                    tNDCG += log(2)/log(idx+2)
                        
                
        print("[%.1f s]After %d epochs and %d iterations, hit@10 and NDCG@10 is %f and %f"%(time()-t1,i,j, hits/6040.0, tNDCG/6040.0))
        #with tf.device('/cpu'):
        #    if i%10 == 0:
        #        saver = tf.train.Saver()
        #        saver.save(sess, "Model/model.ckpt", global_step = i)
                    
                        
                    
                    

