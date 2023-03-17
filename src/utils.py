from jax import random, vmap
import jax.numpy as np
import gym

def keyGen(key, n_subkeys):
	
	keys = random.split(key, n_subkeys + 1)
	
	return keys[0], (k for k in keys[1:])

def print_metrics(phase, duration, t_losses, v_losses = [], batch_range = [], lr = [], epoch = []):
	
	if phase == "batch":
		
		s1 = '\033[1m' + "Batches {}-{} in {:.2f} seconds, learning rate: {:.5f}" + '\033[0m'
		print(s1.format(batch_range[0], batch_range[1], duration, lr))
		
	elif phase == "epoch":
		
		s1 = '\033[1m' + "Epoch {} in {:.1f} minutes" + '\033[0m'
		print(s1.format(epoch, duration / 60))
		
	s2 = """  Training loss {:.4f}"""
	print(s2.format(t_losses.mean()))

	if phase == "epoch":

		s3 = """  Validation loss {:.4f}\n"""
		print(s3.format(v_losses.mean()))

def write_metrics_to_tensorboard(writer, t_losses, v_losses, epoch):

	writer.scalar('VAE loss (train)', t_losses['total'].mean(), epoch)
	writer.scalar('cross entropy (train)', t_losses['cross_entropy'].mean(), epoch)
	writer.scalar('mse (train)', t_losses['mse'].mean(), epoch)
	writer.scalar('KL (train)', t_losses['kl'].mean(), epoch)
	writer.scalar('KL prescale (train)', t_losses['kl_prescale'].mean(), epoch)
	writer.scalar('VAE loss (validation)', v_losses['total'].mean(), epoch)
	writer.scalar('cross entropy (validation)', v_losses['cross_entropy'].mean(), epoch)
	writer.scalar('KL (validation)', v_losses['kl'].mean(), epoch)
	writer.scalar('KL prescale (validation)', v_losses['kl_prescale'].mean(), epoch)
	writer.flush()

def forward_pass_model(model_vae, params_vae, data, state_myo, args, key):

	def apply_model(model_vae, params_vae, data, A, gamma, state_myo, key):

		return model_vae.apply({'params': {'encoder': params_vae['params']['encoder']}}, data, params_vae['decoder'], A, gamma, state_myo, key)

	batch_apply_model = vmap(apply_model, in_axes = (None, None, 0, None, None, None, 0))

	# construct the dynamics of each loop from the parameters
	A, gamma = construct_dynamics_matrix(params_vae['decoder'])

	# create a subkey for each example in the batch
	batch_size = data.shape[0]
	subkeys = random.split(key, batch_size)

	# apply the model
	output = batch_apply_model(model_vae, params_vae, data, A, gamma, state_myo, subkeys)

	# store the original and reconstructed images in the model output
	output['input_images'] = original_images(data, args)
	output['output_images'] = reconstructed_images(output['pen_xy'], output['pen_down_log_p'], args)

	return output