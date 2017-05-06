#*******************************************************************
#   Principles of Brain Computation, SS17
#
#   Template script for Exercise 3
#
#       Robert Legenstein, April 2017
#
#*******************************************************************
# At the beginning we import the necessary Python packages
import nest
from numpy import *       # for numerical operations
from pylab import *       # for plotting (matplotlib)
from pobc_utils import *       # for generating poisson spike trains

nest.set_verbosity("M_WARNING") # surpress too much text output

DT = 0.1       # The time step of the simulation [msec]

def generate_stimulus(nchannels, Rs, jitter, Rbase, Tsim):
    # used by construct_input_population
    if Rs == 0.0:
        Soccur = array([])
    else:
        Soccur = poisson_generator(Rs, t_stop = Tsim*1000)/1000.0
    spikes = []
    for i in range(nchannels):
        s = append(poisson_generator(Rbase, t_stop = Tsim*1000)/1000.0,Soccur+jitter*random.randn(len(Soccur)))
        s.sort()
        s*=1000.0  # times in ms for NEST
        # round to simulation precision
        s *= 10
        s = s.round()+1.0
        s = s/10.0
        spikes.append(s)
    return spikes, Soccur

def generate_stimulus_sequence(nchannels, Rs, jitter, Rbase, Tsim):
    #  used by construct_input_population
    if Rs == 0.0:
        Soccur = array([])
    else:
        Soccur = poisson_generator(Rs, t_stop = Tsim*1000)/1000.0
    spikes = []
    #inp_neuron = SpikingInputNeuron()
    for i in range(nchannels):
        s = append(poisson_generator(Rbase, t_stop = Tsim*1000)/1000.0,i*0.001+Soccur+jitter*random.randn(len(Soccur)))
        s.sort()
        s*=1000.0  # times in ms for NEST
        # round to simulation precision
        s *= 10
        s = s.round()+1.0
        s = s/10.0
        spikes.append(s)
    return spikes, Soccur


def construct_input_population(Nin, jitter, Tsim, sequence):
    # This is a hack.
    # Because in Nest, one cannot connect spike generators with other
    # neurons with STDP synapses, we need to first connect them to a
    # pool of iaf_psc_exp neurons which are then serving as the input pool
    # The pool will produce approximately Poissonian spike trains with rate Rin 
    # Nin...number of input neurons
    # jitter...jitter of population spikes
    # Tsim.....Total simulation time
    # sequence....if True, the stimulus will be shifted in neuron i by i msec
    # Returns:
    # spike_generators...the spike generators' GIDs
    # input_neurons...the input neurons' GIDs
    
    # create input population
        if sequence:
           inp_spikes, s_occur = generate_stimulus_sequence(int(Nin/2), 2.0, jitter, 8.0, Tsim)
        else:
           inp_spikes, s_occur = generate_stimulus(int(Nin/2), 2.0, jitter, 8.0, Tsim)
        inp_spikes_2, s_occur_2 = generate_stimulus(int(Nin/2), 0.0, 0e-3, 8.0, Tsim)
        
        inp_spikes += inp_spikes_2
        
        spike_generators = nest.Create("spike_generator", Nin)
        for (sg, sp) in zip(spike_generators, inp_spikes):
                sp = sp[sp>0]
                nest.SetStatus([sg],{'spike_times': sp})

        input_neurons = nest.Create("iaf_psc_delta",Nin)
        # Choose threshold very close to resting potential so that each spike in a Poisson generator
        # elicits one spike in the corresponding input neuron
        Vresting = -60.0
        nrn_params =     {"V_m": Vresting,     # Membrane potential in mV
                      "E_L": Vresting,     # Resting membrane potential in mV
                      "C_m": 1.0e4/40,           # Capacity of the membrane in pF
                      "tau_m": 0.5,      # Membrane time constant in ms
                      "V_th": -59.9999,     # Spike threshold in mV
                      "V_reset": Vresting, # Reset potential of the membrane in mV
                      "t_ref": .2   # refractory time in ms
                      }
        nest.SetStatus(input_neurons,nrn_params)
        # Connect Poisson generators to input neurons "one-to-one"
        nest.Connect(spike_generators,input_neurons,{'rule':'one_to_one'},syn_spec={'weight':0.1})
        return spike_generators, input_neurons


def perform_simulation(sequence, jitter=0.0, alpha=1.1, Wmax_fact=2, Tsim=200.0, W = 20.0e2):
    """
    Performs the network simulation.
    sequence...If True, stimulus in input population will be sequential
    jitter...Jitter on input population events
    alpha....Scaling factor of negative STDP window size A- = -alpha*A+
    W........Initial weight of synapses
    Wmax_fact.....Maximal synaptic weight is given by Wmax = W * Wmax_fact
    Tsim.....Simulation time
    """
    
        N = 200               # number of input neurons     
        
        #########################################
        # create any neurons, recorders etc. here
        #########################################
        
        
        # the follwoing creates N input neurons and sets their spike trains during simulation
        spike_generators,input_neurons = construct_input_population(N, jitter, Tsim, sequence)
        
        #########################################
        # Connect nodes, simulate
        #########################################
        
        
        # To extract spikes of input neuons as a list of numpy-arrays, use the
        # following function provided in nnb_utils:
        spikes_in = get_spike_times(THE RECORDER THAT RECORDS SPIKES FROM INPUT_NEURONS)
                
        return spikes, weight_evolution, spikes_in

def plot_raster(spikes,tmax):
    """
    Spike raster plot for spikes in 'spikes' up to time tmax [in sec]
    spikes[i]: spike times of neuron i in seconds
    """
    i = 0
    for spks in spikes:
        sp = spks[spks<tmax]
        ns = len(sp)
        plot(sp,i*ones(ns),'b.')
        i=i+1
   

def plot_figures(fig1,fig2, spikes, weights, inp_spikes, Tsim, filename_fig1, filename_fig2, Tmax_spikes=25):
    """
    This function plots two figures for analysis of results
    fig1,fig2....figure identifiers
    spikes.......spikes of the output neuron
    weights......recorded weights over time (column t is the weight vector at recording time index t
                 The function assumes that weights are recorded every second.
    inp_spikes...spikes of input neurons as list of numpy arrays (use get_spike_times)
    Tsim.........simulation time
    filename_figX...filenames of figures to save figures to file
    Tmax_spikes.....Computation of spike cross-correlations may take some time
                    Use Tmax_spikes to compute the cc only over time (0,Tmax_spike)
    """
    # crop spike times in order to save time during convolution:
    Nin = len(weights)
    Nin2 = int(Nin/2)
    spikes = spikes[spikes<Tmax_spikes]
    for i in range(inp_spikes.__len__()):
        inp_spikes[i] = inp_spikes[i][inp_spikes[i]<Tmax_spikes]

    f = figure(fig1, figsize = (8,3.6   ))
    f.subplots_adjust(top= 0.89, left = 0.09, bottom = 0.15, right = 0.93, hspace = 0.30, wspace = 0.40)
    
    ax = subplot(1,2,1)
    imshow(weights, aspect = 'auto')
    ylim(0,Nin+1)
    xlabel('time [sec]')
    colorbar()
    ylabel('synapse id.')
    text(-0.19, 1.07, 'A', fontsize = 'large', transform = ax.transAxes)
    
    ax = subplot(1,2,2)
    mean_up = mean(weights[0:Nin2,:], axis = 0)
    mean_down = mean(weights[Nin2:Nin,:], axis = 0)
    std_up = std(weights[0:Nin2,:], axis = 0)
    std_down = std(weights[Nin2:Nin,:], axis = 0)
    plot(linspace(0,Tsim,len(mean_up)), mean_up, color = 'b')
    plot(linspace(0,Tsim,len(mean_down)), mean_down, color = 'r')
    errorbar(linspace(0, Tsim, len(mean_up))[::20], mean_up[::20], std_up[::20], fmt = 'b.')
    errorbar(linspace(0, Tsim, len(mean_down))[::20], mean_down[::20], std_down[::20], fmt = 'r.')
    xlabel('time [sec]')
    ylabel('avg. syn. weight')
    text(-0.19, 1.07, 'B', fontsize = 'large', transform = ax.transAxes)
    
    savefig(filename_fig1)       

    f = figure(fig2, figsize = (8,8))
    f.subplots_adjust(top= 0.93, left = 0.09, bottom = 0.12, right = 0.95, hspace = 0.40, wspace = 0.40)
    ax = subplot(2,2,1)
    corr = avg_cross_correlate_spikes(inp_spikes[0:Nin2], 200, binsize = 5e-3, corr_range = (-100e-3,100e-3))
    plot(arange(-100e-3,101e-3, 5e-3), corr, marker = 'o')
    xlim(-100e-3,100e-3)
    xticks(list(arange(-0.1,0.101,0.05)), [ '-0.1', '-0.05', '0', '0.05', '0.1' ] )
    xlabel('time lag [sec]')
    axvline(0.0)
    title('input correl. first group', fontsize = 15)
    ylabel('counts/bin')
    text(-0.19, 1.07, 'A', fontsize = 'large', transform = ax.transAxes)

    
    
    ax = subplot(2,2,2)
    corr = avg_cross_correlate_spikes(inp_spikes[Nin2:Nin], 200, binsize = 5e-3, corr_range = (-100e-3,100e-3))
    plot(arange(-100e-3,101e-3, 5e-3), corr, marker = 'o')
    xlim(-100e-3,100e-3)
    xticks(list(arange(-0.1,0.101,0.05)), [ '-0.1', '-0.05', '0', '0.05', '0.1' ] )
    xlabel('time lag [sec]')
    ylabel('counts/bin')
    title('input correl. second group', fontsize = 15)
    axvline(0.0)
    text(-0.19, 1.07, 'B', fontsize = 'large', transform = ax.transAxes)

    
    ax = subplot(2,2,3)
    corr = avg_cross_correlate_spikes_2sets(inp_spikes[0:Nin2], [spikes], binsize = 5e-3, corr_range = (-100e-3,100e-3))
    plot(arange(-100e-3,101e-3, 5e-3), corr, marker = 'o')
    xlim(-100e-3,100e-3)
    xticks(list(arange(-0.1,0.101,0.05)), [ '-0.1', '-0.05', '0', '0.05', '0.1' ] )
    xlabel('time lag [sec]')
    title('input-output correl. first group', fontsize = 15)
    ylabel('counts/bin')
    axvline(0.0)
    text(-0.19, 1.07, 'C', fontsize = 'large', transform = ax.transAxes)

    
    ax = subplot(2,2,4)
    corr = avg_cross_correlate_spikes_2sets(inp_spikes[Nin2:Nin], [spikes], binsize = 5e-3, corr_range = (-100e-3,100e-3))
    plot(arange(-100e-3,101e-3, 5e-3), corr, marker = 'o')
    xlim(-100e-3,95e-3)
    xticks(list(arange(-0.1,0.101,0.05)), [ '-0.1', '-0.05', '0', '0.05', '0.1' ] )
    title('input-output correl. second group', fontsize = 15)
    xlabel('time lag [sec]')
    ylabel('counts/bin')
    axvline(0.0)
    text(-0.19, 1.07, 'D', fontsize = 'large', transform = ax.transAxes)

    savefig(filename_fig2)
