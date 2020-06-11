#Copyright (C) 2018  Seyed Mehran Kazemi, Licensed under the GPL V3; see: <https://www.gnu.org/licenses/gpl-3.0.en.html>
from tensor_factorizer import *
from reader import *
import numpy as np
import gensim
class TransE(TensorFactorizer):

	def __init__(self, params, dataset="dmda"):
		TensorFactorizer.__init__(self, model_name="TransE", loss_function="margin", params=params, dataset=dataset)

	def setup_weights(self):
		# 初始化实体和关系向量，并做归一化处理
		sqrt_size = 6.0 / math.sqrt(self.params.emb_size)
		# self.rel_emb = tf.get_variable(name="rel_emb", initializer=tf.random_uniform(shape=[self.num_rel, self.params.emb_size], minval=-sqrt_size, maxval=sqrt_size))
		self.ent_emb = tf.get_variable(name="ent_emb", initializer=tf.random_uniform(shape=[self.num_ent, self.params.emb_size], minval=-sqrt_size, maxval=sqrt_size))
		new_model = gensim.models.Word2Vec.load('w2v.model')
		vocab = ['pathogenic', 'host', 'group', 'locate', 'genus', 'family', 'order', 'class']
		vector_dim = 100

		# with open("./dmda/entities.dict", 'r', encoding="utf8") as df:
		# 	st = []
		# 	for li in df.readlines():
		# 		li = li.strip('\n')
		# 		st.append(li.split('\t'))
		#
		rel_embedding_matrix = np.zeros((len(vocab), vector_dim))
		# ent_embedding_matrix = np.zeros((len(st), vector_dim))

		for i in range(len(vocab)):
			embedding_vector = new_model.wv[vocab[i]]
			if embedding_vector is not None:
				rel_embedding_matrix[i] = embedding_vector
		#
		# for i in range(len(st)):
		# 	try:
		# 		embedding_vector = new_model.wv[st[i]]
		# 		if embedding_vector is not None:
		# 			ent_embedding_matrix[i] = embedding_vector
		# 	except:
		# 		ent_embedding_matrix[i] = np.ones((1, 300), dtype=int)

		rel_saved_embeddings = tf.constant(rel_embedding_matrix, dtype=tf.float32)
		# ent_saved_embeddings = tf.constant(ent_embedding_matrix, dtype=tf.float32)
		# self.ent_emb = tf.Variable(name="ent_emb",initial_value=ent_saved_embeddings, trainable=False)
		self.rel_emb = tf.Variable(initial_value=rel_saved_embeddings, trainable=False)
		self.var_list = [self.rel_emb, self.ent_emb]

	def define_regularization(self):
		self.regularizer = tf.reduce_sum(tf.nn.relu(tf.square(tf.norm(self.ent_emb, axis=1, ord=2)) - 1.0))

	def gather_train_embeddings(self):
		# 用一个一维的数组，将张量中对应索引的向量提取出来
		self.ph_emb = tf.gather(self.ent_emb, self.ph)
		# print(self.ph)
		# print("111111111111", self.ent_emb, self.ph)
		self.pt_emb = tf.gather(self.ent_emb, self.pt)
		self.nh_emb = tf.gather(self.ent_emb, self.nh)
		self.nt_emb = tf.gather(self.ent_emb, self.nt)
		self.r_emb  = tf.gather(self.rel_emb, self.r)
		# print("111111111111",self.r_emb,self.r)

	def gather_test_embeddings(self):
		self.h_emb = tf.gather(self.ent_emb, self.head) 
		self.r_emb = tf.gather(self.rel_emb, self.rel) 
		self.t_emb = tf.gather(self.ent_emb, self.tail)

	def create_train_model(self):
		self.pos_dissims = tf.norm(self.ph_emb + self.r_emb - self.pt_emb, axis=1, ord=self.params.p_norm)
		self.neg_dissims = tf.norm(self.nh_emb + self.r_emb - self.nt_emb, axis=1, ord=self.params.p_norm)

	def create_test_model(self):
		self.dissims = tf.norm(self.h_emb + self.r_emb - self.t_emb, axis=1, ord=self.params.p_norm)
