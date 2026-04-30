

import sys
sys.path.append('../')

from low_rank_rnns.modules_connectivity_fullRank import *
from low_rank_rnns import TF, stats, plotting as plot, helpers, rankone, ranktwo, clustering

from matplotlib import pyplot as plt
# %matplotlib notebook
import random
import numpy as np

import torch
from scipy.io import savemat


size = 400
noise_std = 1e-2
alpha = .1
lr = 1e-3

input_size = 3
output_size = 1


for i in range(0,50):

	x_train, y_train, mask_train, cohAll_train, x_val, y_val, mask_val, cohAll_val = TF.generate_checker_data(10000)



	# local connections are continuous 
	def create_local_conn(Inum, ratio, n_neurons):

	    mask_rec = torch.zeros(n_neurons, n_neurons)
	    
	    mask_rec[:,n_neurons-Inum:] = -1
	    
	    mask_rec[:,:n_neurons-Inum] = ratio    

	    return mask_rec



	def create_area_conn(Inum, ratio, n_neurons): 
	    mask_rec = torch.zeros(n_neurons, n_neurons)
	    
	    mask_rec[:,n_neurons-Inum:] = 0
	    mask_rec[n_neurons-Inum:,:] = 0
	    
	    # Define the size of the matrix
	    rows, cols = n_neurons-Inum, n_neurons-Inum
	    
	    # Create a zero matrix
	    matrix = torch.zeros(rows, cols)
	    
	    # Calculate the number of entries to set to 1
	    num_entries = int(ratio * rows * cols)
	    
	    # Randomly select indices
	    indices = torch.randperm(rows * cols)[:num_entries]
	    
	    # Set the selected entries to 1
	    matrix.view(-1)[indices] = 1
	    
	    mask_rec[:n_neurons-Inum,:n_neurons-Inum] = matrix

	    return mask_rec








	n_neurons = 400
	n_inputs = 3
	n_outputs = 1


	mask_rec = torch.zeros(n_neurons, n_neurons)
	mask_in = torch.zeros(n_inputs, n_neurons)
	mask_out = torch.ones(n_neurons, n_outputs)


	mask_rec[:100,0:100] = create_local_conn(20,0.2,100)
	mask_rec[100:200,100:200] = create_local_conn(20,0.2,100)
	mask_rec[200:300,200:300] = create_local_conn(20,0.2,100)
	mask_rec[300:400,300:400] = create_local_conn(20,0.2,100)


	connStrength = 0.0357


	mask_rec[:100,100:200] = create_area_conn(20,connStrength,100)



	mask_rec[100:200,0:100] = create_area_conn(20,connStrength,100)

	mask_rec[100:200,200:300] = create_area_conn(20,connStrength,100)

	mask_rec[200:300,100:200] = create_area_conn(20,connStrength,100)



	mask_rec[200:300,300:400] = create_area_conn(20,connStrength,100)
	mask_rec[300:400,200:300] = create_area_conn(20,connStrength,100)


	mask_rec[0:100,200:300] = create_area_conn(20,connStrength,100)





	mask_in[0,:30] = 1
	mask_in[1:,:10] = 1

	mask_in[1:,100:130] = 1
	mask_in[0,100:110] = 1


	mask_out[:300,:] = 0
	mask_out[380:,:] = 0



	wi_mask = mask_in
	wo_mask = mask_out
	wrec_mask = mask_rec.t()
	print(torch.sum(wrec_mask))

	# temp3 = torch.normal(0,1 / sqrt(size),size = (400,400))
	# wrec_init = (np.abs(temp3)*wrec_mask)

	# net = FullRankRNN(3, size, 1, noise_std, alpha, train_wi=True, train_wo = True, train_h0=True, 
	#                   wrec_mask = wrec_mask, wi_mask = wi_mask, wo_mask = wo_mask, wrec_init = wrec_init,
	#                  b_init = None, add_biases = False)

	net = FullRankRNN(3, size, 1, noise_std, alpha, train_wi=True, train_wo = True, train_h0=True, 
	                  wrec_mask = wrec_mask, wi_mask = wi_mask, wo_mask = wo_mask,
	                 b_init = None, add_biases = False)




	# after trained, recurrent connectivity are all zero
	# after trained, bias term will make test accuracy very low
	net.non_linearity = torch.relu
	net.out_non_linearity = torch.relu

	# net.out_non_linearity = torch.eye

	train(net, x_train, y_train, mask_train, n_epochs=20, lr=lr, batch_size=100, 
	      mask_gradients = False, keep_best=True, cuda=True, early_stop=0.25, clip_gradient = 1)

	x_val, y_val, mask_val = map_device([x_val, y_val, mask_val], net)


	loss, acc, out, decisions, target_decisions, traj = TF.test_checker(net, x_val, y_val, mask_val)

	print(f'loss={loss:.3f}, acc={acc:.3f}')
	print('model: ', i, "finished")


	out = out.cpu()
	y_val = y_val.cpu()
	x_val = x_val.cpu()


	traj = traj.cpu()
	decisions = decisions.cpu()

	traj1 = traj.detach().numpy()

	traj1 = np.maximum(traj1,0)

	decisions1 = decisions.detach().numpy()

	tfRL = traj1[np.logical_and(decisions1 == -1, cohAll_val > 0),:,:]
	tfRR = traj1[np.logical_and(decisions1 == 1, cohAll_val > 0),:,:]
	tfGL = traj1[np.logical_and(decisions1 == -1, cohAll_val < 0),:,:]
	tfGR = traj1[np.logical_and(decisions1 == 1, cohAll_val < 0),:,:]

	trajLow = np.zeros((4, tfRL.shape[1], tfRL.shape[2]))
	trajLow[0,:,:] = np.mean(tfRL,axis = 0)
	trajLow[1,:,:] =np.mean(tfRR,axis = 0)
	trajLow[2,:,:] =np.mean(tfGL,axis = 0)
	trajLow[3,:,:] =np.mean(tfGR,axis = 0)

	area = np.arange(0,400)

	dim = trajLow.shape
	firingRatesAverage = np.zeros((400,2,2,dim[1]))

	firingRatesAverage[:,0,0,:] = (trajLow[0,:,area])
	firingRatesAverage[:,0,1,:] = (trajLow[1,:,area])
	firingRatesAverage[:,1,0,:] = (trajLow[2,:,area])
	firingRatesAverage[:,1,1,:] = (trajLow[3,:,area])

	if acc > 0.9: 
		mdic = {"firingRatesAverage": firingRatesAverage}
		savemat(f'../fr/4AreasA3_{i}.mat', mdic)
		torch.save(net.state_dict(), f'../models/4AreasA3_{i}.pt')						