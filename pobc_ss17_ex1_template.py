#*******************************************************************
#   PoBC, SS17
#
#   Template script for Task 1B 
#
#   A leaky integrate-and-fire neuron is created.
#   It gets a spike input at time t=55 ms 
#   There is also an optional step current input at time t=50 ms that forces the neuron to spike
#   There is also an optional step current input at time t=50 ms that forces the neuron to spike
#
#
#
#*******************************************************************
# At the beginning we import the necessary Python packages
from numpy import *       # for numerical operations
import pylab       # for plotting (matplotlib)
import nest
import nest.voltage_trace
import nest.raster_plot

pylab.close('all')


###########################################
#  Parameter for the LIF neuron
###########################################
Rm = 10.0  # [MOhms]
Cm = 2000.  # [pF]
tau_m = Rm*Cm/1000.0  # membrane time constant [ms]
tau_s = 10.0     # synaptic time constant [ms]
Trefract = 10.   # The refractory period of the LIF neuron [ms]
Vthresh = -45.   # The threshold potential of the LIF neuron [mV]
Vresting = -60.  # The resting potential of the LIF neuron [mV]

nrn_parameter_dict = {"V_m": Vresting,     # Membrane potential in mV
                      "E_L": Vresting,     # Resting membrane potential in mV
                      "C_m": Cm,           # Capacity of the membrane in pF
                      "tau_m": tau_m,      # Membrane time constant in ms
                      "t_ref": Trefract,   # Duration of refractory period in ms
                      "V_th": Vthresh,     # Spike threshold in mV
                      "V_reset": Vresting, # Reset potential of the membrane in mV

                      "tau_syn_ex": tau_s, # Time constant of the excitatory synaptic current in ms
                      "I_e": 0.0           # No constant external input current
                      }
# The other neuron parameters have default values.


# Reset the NEST Simulator
nest.ResetKernel()

###################################
# Create nodes
###################################

# Create the IAF neuron, see http://www.nest-simulator.org/cc/iaf_psc_exp/
neuron = nest.Create('iaf_psc_exp', 1, nrn_parameter_dict)

# Create inputs
t_spike_input = 55.
t_step = 50.
step_duration = 0.5
step_amplitude = 60790.
# step_amplitude = 1.

#sine = nest.Create('ac_generator',1,{'amplitude':100.0,'frequency':2.0})
spike_gen = nest.Create("spike_generator", params={"spike_times": array([t_spike_input])})
step_gen = nest.Create('step_current_generator')
nest.SetStatus(step_gen, {'amplitude_times': array([t_step, t_step+step_duration]),
                          'amplitude_values': array([step_amplitude, 0.])})
# step_amplitude is in [pA]
# needs a large value to produce a spike

# Create voltmeter and spike recorder
voltmeter = nest.Create('voltmeter', 1, {'withgid': True})
spike_rec = nest.Create('spike_detector')

###################################
# Connect nodes
###################################

#nest.Connect(sine,neuron)
# Connect spike generator to neuron
# Note: The connection weight is given in [pA]
nest.Connect(spike_gen, neuron, syn_spec={'delay': 2.0})

# Connect current step input step_gen to the neuron
# Note: The current amplitudes as defined above are multiplied with the weight.
nest.Connect(step_gen, neuron, syn_spec={'delay': 2.0})

# Connect voltmeter and spike recorder to neuron
nest.Connect(voltmeter, neuron)
nest.Connect(neuron, spike_rec)

###################################
# Now simulate 
###################################
nest.Simulate(500.0)

###################################
# Analyze results and make plots
###################################


# Extract spikes voltage
spikes = nest.GetStatus(spike_rec)[0]['events']['times']
print(spikes)
# events = nest.GetStatus(spike_rec, 'events')
vm = nest.GetStatus(voltmeter, 'events')[0]['V_m']

# etc.
# Plot results
nest.voltage_trace.from_device(voltmeter)
pylab.show()
# nest.raster_plot.from_device(spike_rec)
# pylab.show()
