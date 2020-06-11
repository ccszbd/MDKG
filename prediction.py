import numpy as np
import pandas as pd
import tensorflow as tf
with open('./dmda/test.txt','r') as tef:
    test = tef.readlines()

x1_test,x2_test,y_test = [],[],[]
for n in range(len(test)):
    x1_test.append(test[n].split('\t')[0])
    x2_test.append(test[n].split('\t')[2].split('\n')[0])
    y_test.append(test[n].split('\t')[1])
# print(x1_test,x2_test,y_test)
x_test = test
# sess = tf.Session()
# saver = tf.train.import_meta_graph('./TransE_weights/dmda/2000.ckpt.meta')
# saver.restore(sess,tf.train.latest_checkpoint('./TransE_weights/dmda/'))
# graph = tf.get_default_graph()
# reader = tf.pywrap_tensorflow.NewCheckpointReader('./complex')
# var_to = reader.get_variable_to_shape_map()

# print([n.name for n in graph.as_graph_def().node])
# xs = graph.get_operation_by_name('rel_emb').outputs[0]
# y = tf.get_collection('ent_emb')
# print(tf.get_collection)
# ys = graph.get_tensor_by_name('rel_emb:0')
# print(graph.get_tensor_by_name('ent_emb:0'))
# print(graph.get_tensor_by_name('rel_emb:0'))
# acc = graph.get_tensor_by_name('accuracy/accuracy:0')
# print(sess.run(y,feed_dict={xs:[[2626265,2564181]]}))

# li = []
# with open("./data/DMDA/relations.dict",'r',encoding='utf8') as en:
#     for line in en.readlines():
#         li.append(line.split('\t')[1])
# with open("./data/relations.vocab",'a',encoding='utf8') as en2:
#     for i in range(len(li)):
#         en2.write(li[i])
# en2.close()