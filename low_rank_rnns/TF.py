
from math import floor
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from low_rank_rnns.modules import loss_mse
from low_rank_rnns.helpers import map_device
import torch

# Task constants
deltaT = 10.
fixation_duration = 100
targets_duration = 2000
stimulus_duration = 500
decision_duration = 500


SCALE = 1
SCALE_CTX = 1
std_default = 1e-1
# decision targets
lo = -1
hi = 1


def setup():
    """
    Call this function whenever changing one of the global task variables (modifies other global variables)
    """
    global fixation_duration_discrete, stimulus_duration_discrete, \
        delay_duration_discrete, decision_duration_discrete, total_duration, stim_begin, stim_end, response_begin, targets_begin
    fixation_duration_discrete = floor(fixation_duration / deltaT)
    targets_duration_discrete = floor(targets_duration / deltaT)
    stimulus_duration_discrete = floor(stimulus_duration / deltaT)
    decision_duration_discrete = floor(decision_duration / deltaT)


    targets_begin = fixation_duration_discrete
    stim_begin = fixation_duration_discrete + targets_duration_discrete - stimulus_duration_discrete
    stim_end = stim_begin + targets_duration_discrete
    response_begin = stim_begin
    total_duration = fixation_duration_discrete + targets_duration_discrete


setup()


def generate_checker_data(num_trials, coherences=None, std=std_default, fraction_validation_trials=0.2,
                        fraction_catch_trials=0., coh_color_spec=None, context_spec=None):
    """
    :param num_trials: int
    :param coherences: list of acceptable coherence values if needed
    :param std: float, if you want to change the std on motion and color inputs
    :param fraction_validation_trials: float, to get a splitted train-test dataset
    :param fraction_catch_trials: float
    :param coh_color_spec: float, to force a specific color coherence, then coherences will be ignored
    :param coh_motion_spec: float, to force a specific motion coherence, then coherences will be ignored
    :param context_spec: in {1, 2}, to force a specific context
    :return: 6 torch tensors, inputs, targets and mask for train and test
    """

    # 3 inputs:
    # 1st input: coh (-1 green to 1 red)
    # 2nd input: red left green right (1 or 0)
    # 3rd input: red right green left (1 or 0)

    # 1 output: choice (-1: left; 1: right)
    if coherences is None:
        coherences = [-0.9, -0.6, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.6, 0.9]

    # inputs_sensory: coh of checkerboard
    inputs_sensory = std * torch.randn((num_trials, total_duration, 1), dtype=torch.float32)
    # inputs_context: targets configuration
    inputs_context = torch.zeros((num_trials, total_duration, 2))
    inputs = torch.cat([inputs_sensory, inputs_context], dim=2)
    # targets: desired ouput 
    targets = torch.zeros((num_trials, total_duration, 1), dtype=torch.float32)
    mask = torch.zeros((num_trials, total_duration, 1), dtype=torch.float32)

    allCoh = np.zeros((num_trials))

    for i in range(num_trials):
        if random.random() > fraction_catch_trials:
            if coh_color_spec is None:
                coh_color = coherences[random.randint(0, len(coherences)-1)]
            else:
                coh_color = coh_color_spec
            
            allCoh[i] = coh_color

            inputs[i, stim_begin:stim_end, 0] += coh_color * SCALE
            if context_spec is None:
                context = random.randint(1, 2)
            else:
                context = context_spec
            if context == 1:
                inputs[i, targets_begin:, 1] = 1. * SCALE_CTX

                targets[i, response_begin:] = lo if coh_color > 0 else hi
            elif context == 2:
                inputs[i, targets_begin:, 2] = 1. * SCALE_CTX

                targets[i, response_begin:] = hi if coh_color > 0 else lo

        mask[i, response_begin:, 0] = 1

    # Split
    split_at = num_trials - int(num_trials * fraction_validation_trials)
    inputs_train, inputs_val = inputs[:split_at], inputs[split_at:]
    targets_train, targets_val = targets[:split_at], targets[split_at:]
    mask_train, mask_val = mask[:split_at], mask[split_at:]

    allCoh_train, allCoh_val = allCoh[:split_at], allCoh[split_at:]

    if fraction_validation_trials > 0:
        return inputs_train, targets_train, mask_train, allCoh_train, inputs_val, targets_val, mask_val, allCoh_val
    elif fraction_validation_trials == 0.:
        return inputs_train, targets_train, mask_train


def generate_interrupt_inputs(num_trials, interrupt_type = 'targets'):


    # inputs_sensory: coh of checkerboard
    inputs_sensory = std * torch.randn((num_trials, total_duration, 1), dtype=torch.float32)
    # inputs_context: targets configuration
    inputs_context = torch.zeros((num_trials, total_duration, 2))    
    inputs = torch.cat([inputs_sensory, inputs_context], dim=2)
    off_time = torch.zeros((num_trials, total_duration, 1), dtype=torch.float32)
    allCoh = np.zeros((num_trials))


    if interrupt_type == 'targets':
        # targets turn on and off
        interrupt_time = [200, 400, 600, 800]

        for i in range(num_trials):
            off_time_each = interrupt_time[random.randint(0, len(interrupt_time)-1)]
            off_time[i] = off_time_each

            context = random.randint(1, 2)
            if context == 1:
                inputs[i, targets_begin:(targets_begin + off_time_each/deltaT), 0] = 1. * SCALE_CTX

            elif context == 2:
                inputs[i, targets_begin:(targets_begin + off_time_each/deltaT), 0] = -1. * SCALE_CTX



    if interrupt_type = 'checkerboard': 
        # checkerboard turn on and off
        coherences = [-0.9, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.9]

        interrupt_time = [600];
        for i in range(num_trials):
            off_time_each = interrupt_time[random.randint(0, len(interrupt_time)-1)]
            off_time[i] = off_time_each

            coh_color = coherences[random.randint(0, len(coherences)-1)]
            allCoh[i] = coh_color
            inputs[i, targets_begin:(targets_begin + off_time_each/deltaT), 1] = coh_color. * SCALE_CTX

    return inputs, off_time, allCoh



def generate_ordered_inputs(trial_repets=1):
    x1, _, _ = generate_checker_data(trial_repets, context_spec=1, coh_color_spec= 0.9, fraction_validation_trials=0.)
    x2, _, _ = generate_checker_data(trial_repets, context_spec=2, coh_color_spec= 0.9, fraction_validation_trials=0.)
    x3, _, _ = generate_checker_data(trial_repets, context_spec=2, coh_color_spec= -0.9, fraction_validation_trials=0.)
    x4, _, _ = generate_checker_data(trial_repets, context_spec=1, coh_color_spec= -0.9, fraction_validation_trials=0.)
    x = torch.cat([x1, x2, x3, x4], dim=0)
    return x


def generate_mante_data_from_conditions(coherences_A, coherences_B, contexts, std=0):
    num_trials = coherences_A.shape[0]
    inputs_sensory = std * torch.randn((num_trials, total_duration, 2), dtype=torch.float32)
    inputs_context = torch.zeros((num_trials, total_duration, 2))
    inputs = torch.cat([inputs_sensory, inputs_context], dim=2)
    targets = torch.zeros((num_trials, total_duration, 1), dtype=torch.float32)
    mask = torch.zeros((num_trials, total_duration, 1), dtype=torch.float32)

    for i in range(num_trials):
        inputs[i, stim_begin:stim_end, 0] += coherences_A[i] * SCALE
        inputs[i, stim_begin:stim_end, 1] += coherences_B[i] * SCALE
        if contexts[i] == 1:
            inputs[i, fixation_duration_discrete:response_begin, 2] = 1. * SCALE_CTX
            targets[i, response_begin:] = hi if coherences_A[i] > 0 else lo
        elif contexts[i] == -1:
            inputs[i, fixation_duration_discrete:response_begin, 3] = 1. * SCALE_CTX
            targets[i, response_begin:] = hi if coherences_B[i] > 0 else lo
        mask[i, response_begin:, 0] = 1
    return inputs, targets, mask

#################### Tian added this 
def generate_mante_data_from_4conditions(coherences_B, contexts, std=0):
    num_trials = coherences_B.shape[0]
    inputs_sensory = std * torch.randn((num_trials, total_duration, 1), dtype=torch.float32)
    inputs_context = torch.zeros((num_trials, total_duration, 2))
    inputs = torch.cat([inputs_sensory, inputs_context], dim=2)
    targets = torch.zeros((num_trials, total_duration, 1), dtype=torch.float32)
    mask = torch.zeros((num_trials, total_duration, 1), dtype=torch.float32)

    for i in range(num_trials):
        inputs[i, stim_begin:stim_end, 0] += coherences_B[i] * SCALE
        if contexts[i] == 1:
            inputs[i, fixation_duration_discrete:response_begin, 1] = 1. * SCALE_CTX
            targets[i, response_begin:] = hi if coherences_B[i] > 0 else lo
        elif contexts[i] == -1:
            inputs[i, fixation_duration_discrete:response_begin, 2] = 1. * SCALE_CTX
            targets[i, response_begin:] = hi if coherences_B[i] > 0 else lo
        mask[i, response_begin:, 0] = 1
    return inputs, targets, mask
####################

def accuracy_checker(targets, output):
    good_trials = (targets != 0).any(dim=1).squeeze()  # remove catch trials
    target_decisions = torch.sign(targets[good_trials, response_begin:, :].mean(dim=1).squeeze())
    decisions = torch.sign(output[good_trials, response_begin:, :].mean(dim=1).squeeze())
    return (target_decisions == decisions).type(torch.float32).mean(), decisions, target_decisions


def test_checker(net, x, y, mask):
    x, y, mask = map_device([x, y, mask], net)
    with torch.no_grad():
        output, traj = net.forward(x, return_dynamics=True)
        loss = loss_mse(output, y, mask)
        acc, decisions, target_decisions = accuracy_checker(y, output)
    return loss.item(), acc.item(), output, decisions, target_decisions, traj


def psychometric_matrices(net, cmap='gray', figsize=(4, 8), axes=None, coherences=None, colorbar=None):
    n_trials = 10
    if coherences is None:
        coherences = np.arange(-5, 6, 2)

    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        # fig2, ax2 = plt.subplots(figsize=figsize)

    for ctx in (0, 1):
        mat = np.zeros((len(coherences), len(coherences)))
        for i, coh1 in enumerate(coherences):
            for j, coh2 in enumerate(coherences):
                inputs_sensory = std_default * torch.randn((n_trials, total_duration, 2), dtype=torch.float32)
                inputs_context = torch.zeros((n_trials, total_duration, 2))
                inputs = torch.cat([inputs_sensory, inputs_context], dim=2)
                inputs[:, stim_begin:stim_end, 0] += coh1 * SCALE
                inputs[:, stim_begin:stim_end, 1] += coh2 * SCALE
                inputs[:, fixation_duration_discrete:response_begin, 2+ctx] = 1. * SCALE_CTX
                output = net.forward(inputs)
                decisions = torch.sign(output[:, response_begin:, :].mean(dim=1).squeeze())
                mat[len(coherences) - j - 1, i] = decisions.mean().item()
        im = axes[ctx].matshow(mat, cmap=cmap, vmin=-1, vmax=1)
        axes[ctx].set_xticks([])
        axes[ctx].set_yticks([])
        axes[ctx].spines['top'].set_visible(True)
        axes[ctx].spines['right'].set_visible(True)
    if colorbar is not None:
        ax_cbar = fig.add_axes([0.9, 0.3, 0.04, 0.4])
        fig.colorbar(im, cax=ax_cbar)
    return axes