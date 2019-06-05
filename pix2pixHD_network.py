import numpy as np
import tensorflow as tf 
from utils import *

class pix2pixHD_network(object):

	def __init__(self, image_A_batch, image_B_batch, batch_size, dropout_rate, args, weights_path=''):    
		# Parse input arguments into class variables
		self.image_A = image_A_batch
		self.image_B = image_B_batch
		self.batch_size = batch_size
		self.dropout_rate = dropout_rate  
		self.WEIGHTS_PATH = weights_path
		self.l1_Weight = 100.0
		self.L2_Weight = 100.0
		self.args = args

	def generator_output(self,image_A_input):
		#G2 structure:
		scope_name = 'gen_G2_conv1'
		self.G2_conv1 = conv(image_A_input, filter_height=3, filter_width=3, num_outputs=self.args.ngf, stride_y=1, stride_x=1, name = scope_name)
		self.G2_conv1_batchnorm = apply_batchnorm(self.G2_conv1, name=scope_name)
		self.G2_conv1_activation = lrelu(self.G2_conv1_batchnorm, lrelu_alpha=0.2, name=scope_name)
		scope_name = 'gen_G2_conv1_downsampling'
		self.G2_conv1_downsampling = conv(self.G2_conv1_activation, filter_height=3, filter_width=3, num_outputs=self.args.ngf*2, stride_y=2, stride_x=2, name = scope_name)
		self.G2_conv1_downsampling_bnorm = apply_batchnorm(self.G2_conv1_downsampling, name=scope_name)
		self.G2_conv1_dwonsampling_activation = lrelu(self.G2_conv1_downsampling_bnorm, lrelu_alpha=0.2, name=scope_name)

		# G1 structure  downsampling
		scope_name = 'gen_G1_pre_block'
		self.G1_pre_block = conv(self.G2_conv1_dwonsampling_activation, filter_height=3, filter_width=3, num_outputs=self.args.ngf*2, stride_y=1,stride_x=1, name=scope_name)
		self.G1_pre_block_activation = lrelu(apply_batchnorm(self.G1_pre_block, name = scope_name), lrelu_alpha=0.2, name=scope_name)

		scope_name = 'gen_G1_downsampling1'
		self.G1_downsampling1 = conv(self.G1_pre_block_activation, filter_height=3, filter_width=3, num_outputs=self.args.ngf*4, stride_y=2, stride_x=2, name=scope_name)
		scope_name = 'gen_G1_downsampling2'
		self.G1_downsampling2 = conv(self.G1_downsampling1, filter_height=3, filter_width=3, num_outputs=self.args.ngf*8, stride_y=2, stride_x=2, name=scope_name)
		scope_name = 'gen_G1_downsampling3'
		self.G1_downsampling3 = conv(self.G1_downsampling2, filter_height=3, filter_width=3, num_outputs=self.args.ngf*16, stride_y=2, stride_x=2, name=scope_name)

		# G1 residual_blocks
		scope_name = 'gen_G1_res_block1'
		self.G1_res_block1 = conv(self.G1_downsampling3, filter_height=3, filter_width=3, num_outputs=self.args.ngf*16, stride_y=1, stride_x=1, name=scope_name)
		self.G1_res_block1_activation = lrelu(apply_batchnorm(self.G1_res_block1, name=scope_name), lrelu_alpha=0.2, name=scope_name)
		scope_name = 'gen_G1_res_block2'
		self.G1_res_block2 = conv(self.G1_res_block1_activation, filter_height=3, filter_width=3, num_outputs=self.args.ngf*16, stride_y=1, stride_x=1, name=scope_name)
		self.G1_res_block2_activation = lrelu(apply_batchnorm(self.G1_res_block2, name=scope_name), lrelu_alpha=0.2, name=scope_name)
		scope_name = 'gen_G1_res_block3'
		self.G1_res_block3 = conv(self.G1_res_block2_activation, filter_height=3, filter_width=3, num_outputs=self.args.ngf*16, stride_y=1, stride_x=1, name=scope_name)
		self.G1_res_block3_activation = lrelu(apply_batchnorm(self.G1_res_block3, name=scope_name), lrelu_alpha=0.2, name=scope_name)
		scope_name = 'gen_G1_res_block4'
		self.G1_res_block4 = conv(self.G1_res_block3_activation, filter_height=3, filter_width=3, num_outputs=self.args.ngf*16, stride_y=1, stride_x=1, name=scope_name)
		self.G1_res_block4_activation = lrelu(apply_batchnorm(self.G1_res_block4, name=scope_name), lrelu_alpha=0.2, name=scope_name)
		scope_name = 'gen_G1_res_block5'
		self.G1_res_block5 = conv(self.G1_res_block4_activation, filter_height=3, filter_width=3, num_outputs=self.args.ngf*16, stride_y=1, stride_x=1, name=scope_name)
		self.G1_res_block5_activation = lrelu(apply_batchnorm(self.G1_res_block5, name=scope_name), lrelu_alpha=0.2, name=scope_name)
		scope_name = 'gen_G1_res_block6'
		self.G1_res_block6 = conv(self.G1_res_block5_activation, filter_height=3, filter_width=3, num_outputs=self.args.ngf*16, stride_y=1, stride_x=1, name=scope_name)
		self.G1_res_block6_activation = lrelu(apply_batchnorm(self.G1_res_block6, name=scope_name), lrelu_alpha=0.2, name=scope_name)


		#G1_upsampling

		scope_name = "gen_G1_upsampling1"
		self.G1_upsampling1 = deconv(self.G1_res_block6_activation, filter_height=3, filter_width=3, num_outputs=self.args.ngf*8, \
			batch_size=self.args.batch_size, stride_y=2, stride_x=2, name=scope_name, padding='SAME')
		scope_name = "gen_G1_upsampling2"
		self.G1_upsampling2 = deconv(self.G1_upsampling1, filter_height=3, filter_width=3, num_outputs=self.args.ngf*4, \
			batch_size=self.args.batch_size, stride_y=2, stride_x=2, name=scope_name, padding='SAME')
		scope_name = "gen_G1_upsampling3"
		self.G1_upsampling3 = deconv(self.G1_upsampling2, filter_height=3, filter_width=3, num_outputs=self.args.ngf*2, \
			batch_size=self.args.batch_size, stride_y=2, stride_x=2, name=scope_name, padding='SAME')
		self.G1_upsampling3_activation = lrelu(apply_batchnorm(self.G1_upsampling3, name=scope_name), lrelu_alpha=0.2, name=scope_name)
		
		# G2 structure
		self.G2_conv2 = tf.add(self.G1_upsampling3_activation, self.G2_conv1_dwonsampling_activation)

		# G2 res_block area
		scope_name = 'gen_G2_res_block1'
		self.G2_res_block1 = conv(self.G2_conv2, filter_height=3, filter_width=3, num_outputs=self.args.ngf*2, stride_y=1, stride_x=1, name=scope_name)
		self.G2_res_block1_activation = lrelu(apply_batchnorm(self.G2_res_block1, name=scope_name), lrelu_alpha=0.2, name=scope_name)
		scope_name = 'gen_G2_res_block2'
		self.G2_res_block2 = conv(self.G2_res_block1_activation, filter_height=3, filter_width=3, num_outputs=self.args.ngf*2, stride_y=1, stride_x=1, name=scope_name)
		self.G2_res_block2_activation = lrelu(apply_batchnorm(self.G2_res_block2, name=scope_name), lrelu_alpha=0.2, name=scope_name)
		scope_name = "gen_G2_res_block3"
		self.G2_res_block3 = conv(self.G2_res_block2_activation, filter_height=3, filter_width=3, num_outputs=self.args.ngf*2, stride_y=1, stride_x=1, name=scope_name)
		self.G2_res_block3_activation = lrelu(apply_batchnorm(self.G2_res_block3, name=scope_name), lrelu_alpha=0.2, name=scope_name)


		scope_name = "gen_G2_upsampling"
		self.G2_upsampling = deconv(self.G2_res_block3_activation, filter_height=3, filter_width=3, num_outputs=self.args.ngf, batch_size=self.args.batch_size, \
			stride_y=2, stride_x=2, name=scope_name, padding='SAME')
		scope_name = 'gen_generator_output_layer'
		self.generator_output_layer = conv(self.G2_upsampling, filter_height=3, filter_width=3, num_outputs=3, stride_y=1, stride_x=1, name = scope_name)
		self.generator_output_layer_activation = tf.nn.tanh(self.generator_output_layer)




		return self.generator_output_layer_activation

	def discriminator_output(self, B_input):
		discrim_input = tf.concat([self.image_A,B_input],3)
		scope_name = 'dis_conv1'
		self.dis_conv1 = lrelu(conv(discrim_input,4,4,64,2,2,padding='SAME',name=scope_name),lrelu_alpha=0.2, name=scope_name)
		scope_name = 'dis_conv2'
		self.dis_conv2 = lrelu(apply_batchnorm(conv(self.dis_conv1,4,4,128,2,2,padding='SAME',name=scope_name),name=scope_name),lrelu_alpha=0.2, name=scope_name)
		scope_name = 'dis_conv3'
		self.dis_conv3 = lrelu(apply_batchnorm(conv(self.dis_conv2,4,4,256,2,2,padding='SAME',name=scope_name),name=scope_name),lrelu_alpha=0.2, name=scope_name)
		scope_name = 'dis_conv4'
		self.dis_conv4 = lrelu(apply_batchnorm(conv(self.dis_conv3,4,4,512,1,1,padding='SAME',name=scope_name),name=scope_name),lrelu_alpha=0.2, name=scope_name)
		scope_name = 'dis_conv5'
		self.dis_conv5 = conv(self.dis_conv4,4,4,1,1,1,padding='SAME',name=scope_name)
		self.dis_out_per_patch = tf.reshape(self.dis_conv5,[self.batch_size,-1])
		return tf.sigmoid(self.dis_out_per_patch)

	def compute_loss(self):
		eps = 1e-12
		fake_B = self.generator_output(self.image_A)
		fake_output_D = self.discriminator_output(fake_B)
		real_output_D = self.discriminator_output(self.image_B)		
		
		self.d_loss = tf.reduce_mean(-(tf.log(real_output_D + eps) + tf.log(1 - fake_output_D + eps)))     
		self.g_loss_l1= self.l1_Weight*tf.reduce_mean(tf.abs(fake_B - self.image_B))
		self.g_loss_gan = tf.reduce_mean(-tf.log(fake_output_D + eps))

		self.g_loss_mse = self.L2_Weight*tf.reduce_mean(tf.square(fake_B-self.image_B))

		return self.d_loss, self.g_loss_l1 + self.g_loss_gan, self.g_loss_l1, self.g_loss_gan, self.g_loss_mse

	def load_initial_weights(self, session):
		# Load the weights into memory: this approach is adopted rather than standard random initialization to allow the
		# flexibility to load weights from a numpy file or other files.
		if self.WEIGHTS_PATH:
			print('loading initial weights from '+ self.WEIGHTS_PATH)
			weights_dict = np.load(self.WEIGHTS_PATH, encoding = 'bytes').item()  
		# else:
		# 	print 'loading random weights'
		# 	weights_dict = get_random_weight_dictionary('pix2pix_initial_weights')
		# Loop over all layer names stored in the weights dict
		for op_name in weights_dict:          
			print(op_name)
			with tf.variable_scope(op_name) as scope:  
				for sub_op_name in weights_dict[op_name]:
	  				data = weights_dict[op_name][sub_op_name]
	  				var = get_scope_variable(name, sub_op_name, shape=[data.shape[0], data.shape[1], data.shape[2], data.shape[3]])
	  				session.run(var.assign(data))
