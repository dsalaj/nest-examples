## PoBC 2017, ex5
## Competitive learning in WTA networks

import matplotlib.pyplot as plt

import nest
import nest.raster_plot
import nest.voltage_trace
import pylab
import numpy as np

#from pobc_utils import get_spike_times, poisson_generator

def get_rate_patterns(N_in, N_pat, Rmax = 50.0, Rvar = 50.0, PLOT=True):
    # generate N_pat rate patterns for the N_in input neurons
    # Rate patterns have spatial Gaussian profiles with a maximum rate of Rmax
    # and a spatial variance defined by Rvar
    # PLOT... if True, the patterns are plotted
    # RETURNS:
    # rates.....A list of input rate patterns. rates[i] is a vector of rates [Hz], one for each input neuron.
    #           rates[i][j] is the rate of the j-th input neuron in pattern i.
    
    step = N_in/N_pat-2  # defines the means of the Gaussian rate profiles
    offs = N_in/N_pat/2+2 # defines the offset of the means
    rates=[]
    idxs = np.array(range(N_in))
    for i in range(N_pat):
      rr = 50.0*np.exp(-(idxs-(offs+i*step))**2/50.0)
      rates.append(rr)
    if PLOT:
        plt.figure()
        leg = []
        idx = 0
        for rr in rates:
            plt.plot(rr)
            leg.append('pattern ' + str(idx))
            idx += 1
        plt.xlabel('input neuron index')
        plt.ylabel('firing rate [Hz]')
        plt.legend(leg)
        plt.title('Input rate patterns')
    return rates

def set_pattern(nodes_inp, rates, pat_idx):
    # Set the rates of the Poisson_generators nodes_inp to rates[pat_idx]
    # and set their time origin to the global simulation time
    for nrn, r in zip(nodes_inp,rates[pat_idx]):
         nest.SetStatus([nrn],{'rate':r})
    # set input times
    nest.SetStatus(nodes_inp,{'origin': nest.GetKernelStatus('time')})
      # Note: When we do consecutive simulations in nest, the internal simulation time
      #       is continued from the last one.
      #       nest.GetKernelStatus('time') returns the current time of the simulation
      #       We use this to set the origin of the input nodes (Poisson generator) such that
      #       the spikes are emitted there relative to this origin (i.e. at correct times)
      #       See also below where we define 'nodes_inp'

def set_random_pattern(nodes_inp, rates):
    # Set the rates of the Poisson_generators nodes_inp to
    # a randomly chosen rate pattern in rates
    # and set their time origin to the global simulation time
    set_pattern(nodes_inp, rates,np.random.randint(len(rates)))

def test_nework(rates,nodes_inp,simtime,Ntest=10):
  # Test the current network in rate input patterns
  # We go through all patterns in 'rates'. Each input pattern is presented Ntest times consecutively
  # rates...........A list of input rate patterns. rates[i] is a vector of rates [Hz], one for each input neuron.
  #                 rates[i][j] is the rate of the j-th input neuron in pattern i.
  # nodes_inp[i]....GID of input neuron i
  # simtime.........length of a single simulation run (one pattern presentation) in [s].
  # Ntest...........Number of simulations per pattern
    
  N_pat = len(rates)
  nest.ResetNetwork()
  for pattern in range(N_pat):
    for ep in range(Ntest):
      # set rates
      #nest.ResetNetwork()
      set_pattern(nodes_inp, rates, pattern)
      # SIMULATE!! -----------------------------------------------------
      nest.Simulate(simtime)

def update_weights(plast_params,spikes_inp, spikes_E, nodes_inp, nodes_E):
# Update the weights of the excitatory neurons based on pre-post pairings
# Depress if there was a post-spike but no pre
# Potentiate if there was pre and post spike
# If A_decay>0 in plast_params, decay all incoming weights to neuron if the
#              the neuron did not spike, but some neuron spiked in the network
# plast_params..Parameters for the updates, see below in main part
# spikes_inp... The spikes of input neurons
# spikes_E..... The spikes of excitatory neurons
# nodes_X...... GIDs of inp and E

      # Unpack Plasticity parameters
      w_max = plast_params['w_max']
      eta   = plast_params['eta']
      A_neg = plast_params['A_neg']
      A_pos = plast_params['A_pos']
      A_decay = plast_params['A_decay']
      
      # Get network events
      events = nest.GetStatus(spikes_inp,'events')[0]
      active_input_neurons = list(events['senders'])
      events = nest.GetStatus(spikes_E,'events')[0]
      active_wta_neurons = list(events['senders'])
      if len(active_wta_neurons): # Update only if there was a network spike
        # slight weight decay
        if A_decay>0:
          conns = nest.GetConnections(nodes_inp, nodes_E)
          w = np.array(nest.GetStatus(conns,'weight'))
          w -= eta*A_decay
          w[w<0.] = 0.
          nest.SetStatus(conns,'weight',w)
        # first depress all incoming weights
        conns = nest.GetConnections(nodes_inp, active_wta_neurons)
        w = np.array(nest.GetStatus(conns,'weight'))
        w = w-eta*A_neg
        w[w<0.] = 0.
        nest.SetStatus(conns,'weight',w)
        # Now potentiate active inputs
        if len(active_input_neurons):
           conns = nest.GetConnections(active_input_neurons, active_wta_neurons)
           w = np.array(nest.GetStatus(conns,'weight'))
           w = w+eta*A_pos
           w[w>w_max] = w_max
           nest.SetStatus(conns,'weight',w)

def update_weights_wdep(plast_params,spikes_inp, spikes_E, nodes_inp, nodes_E):
# Update the weights of the excitatory neurons based on pre-post pairings
# Depress if there was a post-spike but no pre by (w/w_max)^alpha
# Potentiate if there was pre and post spike by ((w_max-w)/w_max)^alpha
# If A_decay>0 in plast_params, decay all incoming weights to neuron if the
#              the neuron did not spike, but some neuron spiked in the network
# plast_params..Parameters for the updates, see below in main part
# spikes_inp... The spikes of input neurons
# spikes_E..... The spikes of excitatory neurons
# nodes_X...... GIDs of inp and E

      # Unpack Plasticity parameters
      w_max = plast_params['w_max']
      eta   = plast_params['eta']
      alpha = plast_params['alpha']
      A_decay = plast_params['A_decay']
      
      # Get network events
      events = nest.GetStatus(spikes_inp,'events')[0]
      active_input_neurons = list(events['senders'])
      events = nest.GetStatus(spikes_E,'events')[0]
      active_wta_neurons = list(events['senders'])
      if len(active_wta_neurons): # Update only if there was a network spike
        # slight weight decay
        if A_decay>0:
          conns = nest.GetConnections(nodes_inp, nodes_E)
          w = np.array(nest.GetStatus(conns,'weight'))
          w -= eta*A_decay
          w[w<0.] = 0.
          nest.SetStatus(conns,'weight',w)
        # Backup active input weights
        if len(active_input_neurons):
           conns_act = nest.GetConnections(active_input_neurons, active_wta_neurons)
           w_act = np.array(nest.GetStatus(conns_act,'weight'))
        # depress all incoming weights
        conns = nest.GetConnections(nodes_inp, active_wta_neurons)
        w = np.array(nest.GetStatus(conns,'weight'))
        w = w - eta*(w/w_max)**alpha
        w[w<0] = 0.
        nest.SetStatus(conns,'weight',w)
        # Undo depression for active inputs
        if len(active_input_neurons):
           nest.SetStatus(conns_act,'weight',w_act)
        # Now potentiate active inputs
        if len(active_input_neurons):
           w = np.array(nest.GetStatus(conns_act,'weight'))
           w = w+eta*((w_max-w)/w_max)**alpha
           w[w>w_max] = w_max
           nest.SetStatus(conns_act,'weight',w)

###########################################
# MAIN
###########################################

_TUNE_NETWORK = False  # if True, we just simulate the network once and check its behavior
                      # no learning
                      # used to set parameters of the network to obtain good WTA behavior
plt.close('all')
simtime = 100. # [ms] # simulation time for the presentation of a single pattern
WDEP = True   # define whether to use a linear weight dependency in the updates

####################
# Network parameters
####################
N_in = 80 # number of input neurons
N_E = 10  # number of excitatory neurons
N_I = 50  # number of inhibitory neurons
N_neurons = N_E + N_I  # total number of neurons
N_pat = 5 # Number of different patterns

# Plasticity parameters for the case of no weight dependency
plast_params_nowdep = {
      'w_max':   ??,       # Max weight of plastic synapses // on the order or tens
      'eta':     ??,       # learning rate
      'A_neg':   ??,       # LTD factor
      'A_pos':   ??,       # LTP factor
      'A_decay': 0.}       # weight decay factor [Not used]

# Plasticity parameters for the case of weight dependency
plast_params_wdep = {
      'w_max':   ??,       # Max weight of plastic synapses // here, it should be relatively high (why?)
      'eta':     ??,       # learning rate
      'alpha':  0.5,       # exponent of weight dependency
      'A_decay': 0.}       # weight decay factor

if not(WDEP):  # Task 5B, no weight dependence
    plast_params = plast_params_nowdep
    Nep = ?? # number of pattern presentations during learning
else:          # Task 5C, weight dependence
    plast_params = plast_params_wdep
    Nep = ?? # number of pattern presentations during learning

# Connection parameters
J_in = ??   # initial strength of Input->E synapses [pA]
J_EI = ??  # strength of E->I synapses [pA]
J_IE = ??  # strength of inhibitory synapses [pA]
J_noise = ??  # strength of synapses from noise input [pA]
rate_noise = 100. # rate of Poission background noise [Hz]

# recording parameters
rec_every = 10    # Record weights every 'rec_every' pattern presentation
Nrec = int(Nep/rec_every)

# Define rates paterns for the input neurons
rates = get_rate_patterns(N_in, N_pat, PLOT=_TUNE_NETWORK)

# Set parameters of the NEST simulation kernel
nest.ResetKernel()
nest.set_verbosity('M_ERROR')  # Do not print stuff during simulation
nest.SetKernelStatus({'print_time': False,
                      'local_num_threads': 11}) # Number of threads used

####################
# Create nodes 
####################
nest.SetDefaults('iaf_psc_exp',
                 {'C_m': 30.0,  
                  'tau_m': 30.0,
                  'I_e': 0.0,
                  'E_L': -70.0,
                  'V_th': -55.0,
                  'tau_syn_ex': 3.0,
                  'tau_syn_in': 2.0,
                  'V_reset': -70.0})

# Create excitatory and inhibitory populations
nodes = nest.Create('iaf_psc_exp', N_neurons)

nodes_E = nodes[:N_E]
nodes_I = nodes[N_E:]

# Create noise input
noise_E = nest.Create('poisson_generator', N_E, {'rate': rate_noise})
noise_I = nest.Create('poisson_generator', N_E, {'rate': rate_noise})

# create spike detectors from excitatory and inhibitory populations
spikes = nest.Create('spike_detector', 3,
                     [{'label': 'inp_spd'},
                      {'label': 'ex_spd'},
                      {'label': 'in_spd'}])
spikes_inp = [spikes[0]]
spikes_E = [spikes[1]]
spikes_I = [spikes[2]]
# Create voltmeter and spike recorder
voltmeter = nest.Create('voltmeter',1)

# create input generators
nodes_inp = nest.Create("poisson_generator", N_in)
nest.SetStatus(nodes_inp,{"rate":0.,"start":60.,"stop":80.})
# Poisson_generator:
# - rate:  firing rate <- we will set this later to the rate in the rate pattern
# - start: start time of firing <- is set to 60 ms
# - stop:  end time of firing <- is set to 80 ms
# - origin: the start and stop times are relative to 'origin' in terms of global nest simulation time
#           we will set the origin later when simulations are performed consecutively


####################
# Connect nodes 
####################

# connect inputs to neurons
nest.Connect(nodes_inp, nodes_E,
                {'rule': 'all_to_all'},
                {'model': 'static_synapse',
                'delay': 1,
                'weight': {'distribution': 'uniform',
                            'low': 2.5 * J_in,
                            'high': 7.5 * J_in}})
# connect e to i
nest.Connect(nodes_E, nodes_I,
                {'rule': 'all_to_all'},
                {'model': 'static_synapse',
                'delay': 0.1,
                'weight': {'distribution': 'uniform',
                            'low': 2.5 * J_EI,
                            'high': 7.5 * J_EI}})
# connect i to e
nest.Connect(nodes_I, nodes_E,
                {'rule': 'all_to_all'},
                {'model': 'static_synapse',
                'delay': 0.1,
                'weight': {"distribution": "normal", "mu": J_IE, "sigma": 0.7 * abs(J_IE)}})

# connect noise generators to neurons
# Each excitatory neuron gets excitatory and inhibitory noise
# this leads to random fluctuations of the membrane potential
nest.CopyModel('static_synapse_hom_w',
               'excitatory_noise',
               {'weight': J_noise, 'delay': 1.})
nest.Connect(noise_E, nodes_E,
             {'rule': 'one_to_one'},
             {'model': 'excitatory_noise'})
nest.CopyModel('static_synapse_hom_w',
               'inhibitory_noise',
               {'weight': -J_noise, 'delay': 1.})
nest.Connect(noise_I, nodes_E,
             {'rule': 'one_to_one'},
             {'model': 'inhibitory_noise'})
# connect recorders
nest.Connect(nodes_inp, spikes_inp)
nest.Connect(nodes_E, spikes_E)
nest.Connect(nodes_I, spikes_I)
nest.Connect(voltmeter,nodes_E)    

##############################
# Test before learning
##############################
test_nework(rates,nodes_inp,simtime)
# Show spikes and voltages
nest.raster_plot.from_device(spikes_E, hist=True)
plt.title('Network spikes before learning')
plt.figure()
nest.voltage_trace.from_device(voltmeter)
plt.title('Membrane potentials before learning')

if _TUNE_NETWORK:
    # plot input spikes
    nest.raster_plot.from_device(spikes_inp, hist=True)
    plt.title('Input spikes')
    # plot membrane potentials for each pattern
    for i in range(N_pat):
      test_nework([rates[i]],nodes_inp,simtime,Ntest=1)
      plt.figure()
      nest.voltage_trace.from_device(voltmeter)
      plt.title('Membrane potentials for input pattern ' + str(i) )

if not(_TUNE_NETWORK):
##############################
# Train
##############################
    # initialize weight recording buffers
    Wrec = [] # Wrec[i] will hold the evolution of weights for the i-th excitatory neuron
    in_conns = [] # in_conns[i] will hold the connection ID-s of incoming synapses for the i-th excitatory neuron  
    for i in range(N_E):
      Wrec.append(np.zeros((Nrec,N_in)))
      in_conns.append(nest.GetConnections(nodes_inp, [nodes_E[i]]))

    rec_idx = 0 # running index, specifying at which slot the weights are stored for recording
    for ep in range(Nep):
      nest.ResetNetwork()
      # set rate patters for input neurons (chosen randomly)
      set_random_pattern(nodes_inp, rates)
      # SIMULATE!! -----------------------------------------------------
      nest.Simulate(simtime)
      # Weight updates
      if WDEP:
          update_weights_wdep(plast_params,spikes_inp, spikes_E, nodes_inp, nodes_E)
      else:
          update_weights(plast_params,spikes_inp, spikes_E, nodes_inp, nodes_E)
      # Record weights
      if np.mod(ep,rec_every)==0:
         for i in range(N_E):
            Wrec[i][rec_idx,:] = np.array(nest.GetStatus(in_conns[i],'weight'))
         if np.mod(ep,10*rec_every)==0:
            nevents = nest.GetStatus(spikes, 'n_events')
            print("WTA spikes: ", nevents[1])
         rec_idx += 1
    # LEARNING IS DONE NOW
    
    # plot weight evolution
    plt.figure()
    for i in range(N_E):
      plt.subplot(1,N_E,i+1)
      plt.imshow(Wrec[i]); plt.colorbar();
    # Plot final weights
    plt.figure()
    for i in range(N_E):
      plt_nrn = [nodes_E[i]]
      conns = nest.GetConnections(nodes_inp, plt_nrn)
      w = np.array(nest.GetStatus(conns,'weight'))
      plt.plot(np.array(range(N_in))+1,w)
      plt.xlabel('input neuron')
      plt.ylabel('weights to WTA neuron')

    # Test after learning
    test_nework(rates,nodes_inp,simtime)
    nest.raster_plot.from_device(spikes_E, hist=True)
    plt.title('Network spikes after learning')
    plt.figure()
    nest.voltage_trace.from_device(voltmeter)
    plt.title('Membrane potentials after learning')

plt.show()

