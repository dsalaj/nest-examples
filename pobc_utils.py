#from numpy import array, log
import numpy
import numpy as np
from pylab import find
import nest


def get_spike_times(spike_rec):
    """
       Takes a spike recorder spike_rec and returns the spikes in a list of numpy arrays.
       Each array has all spike times of one sender (neuron) in units of [sec]
    """
    events = nest.GetStatus(spike_rec)[0]['events']
    min_idx = min(events['senders'])
    max_idx = max(events['senders'])
    spikes = []
    for i in range(min_idx, max_idx + 1):
        idx = find(events['senders'] == i)
        spikes.append(events['times'][idx] / 1000.0)  # convert times to [sec]
    return spikes


def cross_correlate_spikes(s1, s2, binsize, corr_range):
    # Compute cross-correlation between two spike trains
    # The implementation is rather inefficient
    cr_lo = corr_range[0]
    cr_hi = corr_range[1]
    ttt = corr_range[1] - corr_range[0]
    Nbins = np.ceil(ttt / binsize)
    Nbins_h = round(Nbins / 2)
    corr = np.zeros(Nbins + 1)
    s1a = np.append(s1, np.inf)
    for t in s2:
        idx = 0
        while s1a[idx] < t + cr_lo:
            idx += 1
        while s1a[idx] < t + cr_hi:
            idxc = round((t - s1a[idx]) / binsize) + Nbins_h
            corr[idxc] += 1
            idx += 1
    return corr


def avg_cross_correlate_spikes(spikes, num_pairs, binsize, corr_range):
    """
       computes average cross-crrelation between pairs of spike trains in spikes in the
       range defince by corr_range and with bin-size defined by binsize.
    """
    i = np.random.randint(len(spikes))
    j = np.random.randint(len(spikes))
    if i == j:
        j = (i + 1) % len(spikes)
    s1 = spikes[i]
    s2 = spikes[j]
    corr = cross_correlate_spikes(s1, s2, binsize, corr_range)
    for p in range(1, num_pairs):
        i = np.random.randint(len(spikes))
        j = np.random.randint(len(spikes))
        if i == j:
            j = (i + 1) % len(spikes)
        s1 = spikes[i]
        s2 = spikes[j]
        corr += cross_correlate_spikes(s1, s2, binsize, corr_range)
    return corr


def avg_cross_correlate_spikes_2sets(spikes1, spikes2, binsize, corr_range):
    s1 = spikes1[0]
    s2 = spikes2[0]
    corr = cross_correlate_spikes(s1, s2, binsize, corr_range)
    for i in range(1, len(spikes1)):
        for j in range(1, len(spikes2)):
            s1 = spikes1[i]
            s2 = spikes2[j]
            corr += cross_correlate_spikes(s1, s2, binsize, corr_range)
    return corr

def poisson_generator(rate, t_start=0.0, t_stop=1000.0, rng=None):
    """
    Returns a SpikeTrain whose spikes are a realization of a Poisson process
    with the given rate (Hz) and stopping time t_stop (milliseconds).

    Note: t_start is always 0.0, thus all realizations are as if 
    they spiked at t=0.0, though this spike is not included in the SpikeList.

    Inputs:
        rate    - the rate of the discharge (in Hz)
        t_start - the beginning of the SpikeTrain (in ms)
        t_stop  - the end of the SpikeTrain (in ms)
        array   - if True, a numpy array of sorted spikes is returned,
                  rather than a SpikeTrain object.

    Examples:
        >> gen.poisson_generator(50, 0, 1000)
        >> gen.poisson_generator(20, 5000, 10000, array=True)
     
    See also:
        inh_poisson_generator
    """

    if rng == None:
        rng = np.random

    #number = int((t_stop-t_start)/1000.0*2.0*rate)

    # less wasteful than double length method above
    n = (t_stop - t_start) / 1000.0 * rate
    number = numpy.ceil(n + 3 * numpy.sqrt(n))
    if number < 100:
        number = min(5 + numpy.ceil(2 * n), 100)

    number = int(number)
    if number > 0:
        isi = rng.exponential(1.0 / rate, number) * 1000.0
        if number > 1:
            spikes = numpy.add.accumulate(isi)
        else:
            spikes = isi
    else:
        spikes = numpy.array([])

    spikes += t_start
    i = numpy.searchsorted(spikes, t_stop)

    extra_spikes = []
    if i == len(spikes):
        # ISI buf overrun

        t_last = spikes[-1] + rng.exponential(1.0 / rate, 1)[0] * 1000.0

        while (t_last < t_stop):
            extra_spikes.append(t_last)
            t_last += rng.exponential(1.0 / rate, 1)[0] * 1000.0

        spikes = numpy.concatenate((spikes, extra_spikes))

    else:
        spikes = numpy.resize(spikes, (i,))

    return spikes
