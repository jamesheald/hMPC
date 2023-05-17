from jax import random
import jax.numpy as np
from flax.metrics import tensorboard

def keyGen(key, n_subkeys):
	
	keys = random.split(key, n_subkeys + 1)
	
	return keys[0], (k for k in keys[1:])

def stabilise_variance(log_var, var_min = 1e-16):
	"""
	var_min is added to the variances for numerical stability
	"""
	return np.log(np.exp(log_var) + var_min)

def print_metrics(phase, duration, t_losses, v_losses = [], batch_range = [], lr = [], epoch = []):
	
	if phase == "batch":
		
		s1 = '\033[1m' + "Batches {}-{} in {:.2f} seconds, learning rate: {:.5f}" + '\033[0m'
		print(s1.format(batch_range[0], batch_range[1], duration, lr))
		
	elif phase == "epoch":
		
		s1 = '\033[1m' + "Epoch {} in {:.1f} minutes" + '\033[0m'
		print(s1.format(epoch, duration / 60))
		
	s2 = """  Training loss {:.10f}"""
	print(s2.format(t_losses.mean()))

	if phase == "epoch":

		s3 = """  Validation loss {:.10f}\n"""
		print(s3.format(v_losses.mean()))

def create_tensorboard_writer(args):

    # create a tensorboard writer
    # to view tensorboard results, call 'tensorboard --logdir=.' in runs folder from terminal
    writer = tensorboard.SummaryWriter('runs/' + args.folder_name)

    return writer

def write_metrics_to_tensorboard(writer, t_losses, v_losses, epoch):

	writer.scalar('GRU loss (train)', t_losses['total'].mean(), epoch)
	writer.flush()