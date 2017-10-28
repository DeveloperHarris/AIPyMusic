from deepmusic.moduleloader import ModuleLoader
from deepmusic.keyboardcell import keyboardcell
import deepmusic.songstruct as music
import numpy 
import tensorflow as tf 


def build_network(self):
	input_dim = ModuleLoader.batch_builders.get_module().get_input_dim()


	#note data	
	with tf.name_scope('placeholder_inputs'):
		self.inputs = [
			tf.placeholder_inputs(
				tf.float32, 
				[self.args.batch_size, input_dim],
				name='input')
		]


	#targets 88 key binary classification
	with tf.name_scope('placeholder_targets'):
		self.targets = [
			tf.placeholder(
				tf.int32,
				[self.batch_size],
				name='target')
		]


	#hidden state
	with tf.name_scope('placeholder_use_prev'):
		self.use_prev = [
			tf.placeholder(
				tf.bool,
				[],
				name='use_prev')
		]


	#define network
	self.loop_processing = ModuleLoader.loop_processing.build_module(self.args)
	def loop_rnn(prev, i):
		next_input = self.loop_processing(prev)
		return tf.cond(self.prev[i], lambda: next_input, lambda:self.inputs[i]


	#build seq 2 seq
	self.outputs, self.final_state = tf.nn.seq2seq.rnn_decoder(
		decoder_inputs = self.inputs,
		inital_state = None,
		cell = KeyboardCell,
		loop_function = loop_rnn)


	#train

	loss_fct = tf.nn.seq2seq.sequence_loss(
		self.outputs,
		self.targets,
		softmax_loss_function = tf.nn.softmax.cross_entropy_with_logits,
		average_across_timesteps = True,
		average_across_timesteps = True,
		average_across_batches = True)


	#init optimize
	opt = tf.train.AdamOptimizer(
		learning_rate = self.current_learning_rate,
		beta1 = 0.90,
		beta2 = 0.999,
		epsilon = 1e-08
	)

	self.opt_op	= opt.minimize(loss_fct)

