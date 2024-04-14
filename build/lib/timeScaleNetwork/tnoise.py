import numpy as np
from numpy.lib.shape_base import tile


def get_noise( signal_shape, beta=0):
    '''Creates spectrum-sepcified noise.  Beta value deterimines how rapidly the noise increases/decreases with 

    Args:
        signal_shape (iterable): The shape of the output array containing the noise. Noise will be inserted along the last axis.
        beta (float, optional):  The logarithmic slope of the rise/decay in frequency power.

    Returns:
        Numpy Array of the specified shape containing the spectrum-specified noise.
    '''
    # Create a properly scaled frequency magnitude
    target_power_spectrum = np.empty(signal_shape)
    if len(signal_shape) == 1:
        # Strip the first value of fftfreq() to avoid difision by zero error
        target_power_spectrum[0]  = 1
        # Exclude zero Hz to avoid division by zero later
        frequencies = np.fft.fftfreq(signal_shape[-1])[1:]
        # In np.power calculation, must take abs() to avoid fractional powers of negative nubmers, which may result in complex numbers. Sign is multiplied back in after power calculation.
        target_power_spectrum[1:] = np.divide( np.ones(signal_shape[-1]-1), np.sign(frequencies) * np.power( np.abs(frequencies), beta))
    else:
        target_shape = list( signal_shape)
        frequencies  = np.fft.fftfreq(target_shape[-1])[1:] # Exclude zero Hz to avoid division by zero later
        target_shape[-1] = target_shape[-1] - 1             # Exclude zero Hz to avoid division by zero later
        # Duplicate frequencies to match the target shape
        npTile_dims = target_shape
        npTile_dims[-1] = int( npTile_dims[-1] / frequencies.shape[-1])

        target_power_spectrum[:, 0]  = 1
        # In np.power calculation, must take abs() to avoid fractional powers of negative nubmers, which may result in complex numbers. Sign is multiplied back in after power calculation.
        target_power_spectrum[:, 1:] = np.divide( np.ones(target_shape), np.power( np.abs(frequencies) * np.tile( np.abs(frequencies), npTile_dims), beta))
    
    # Randomize the phase, and onvert to time domain
    noise = np.real( np.fft.ifft( np.multiply( target_power_spectrum, np.exp(2.0*np.pi*(0+1j)*np.random.uniform(size=signal_shape[-1])))))
    # Normalize to be between -1 and 1
    return noise / (np.amax( np.abs(noise)))


def get_white( shape):
    '''Creates uniform-frequency-spectrum white noise. Uses :py:func:`~get_noise` with a beta value of 0.

    Args:
        shape (iterable): The shape of the output array containing the noise. Noise will be inserted along the last axis.
        
    Returns:
        Numpy Array of the specified shape containing the spectrum-specified noise.
    '''
    return get_noise( shape, 0)


def get_pink( shape):
    '''Creates decreasing-frequency-spectrum pink noise. Uses :py:func:`~get_noise` with a beta value of 1.

    Args:
        shape (iterable): The shape of the output array containing the noise. Noise will be inserted along the last axis.
        
    Returns:
        Numpy Array of the specified shape containing the spectrum-specified noise.
    '''
    return get_noise( shape, 1)


def get_brown( shape):
    '''Creates decreasing-frequency-spectrum brown noise. Uses :py:func:`~get_noise` with a beta value of 2.

    Args:
        shape (iterable): The shape of the output array containing the noise. Noise will be inserted along the last axis.
        
    Returns:
        Numpy Array of the specified shape containing the spectrum-specified noise.
    '''
    return get_noise( shape, 2)


def get_blue( shape):
    '''Creates increasing-frequency-spectrum blue noise. Uses :py:func:`~get_noise` with a beta value of -1.

    Args:
        shape (iterable): The shape of the output array containing the noise. Noise will be inserted along the last axis.
        
    Returns:
        Numpy Array of the specified shape containing the spectrum-specified noise.
    '''
    return get_noise( shape, -1)


def get_violet( shape):
    '''Creates increasing-frequency-spectrum violet noise. Uses :py:func:`~get_noise` with a beta value of -2.

    Args:
        shape (iterable): The shape of the output array containing the noise. Noise will be inserted along the last axis.
        
    Returns:
        Numpy Array of the specified shape containing the spectrum-specified noise.
    '''
    return get_noise( shape, -2)

# TODO: Add Grey noise, which is frequency scaled to match a psycho-acustic equal loudenss curve


def get_color( shape, color):
    '''Helper funciton to allow easy selection of the color of noise to use.

    Args:
        shape (iterable): The shape of the output array containing the noise. Noise will be inserted along the last axis.
        color (str):      String designating the color of noise to use.
        
    Returns:
        Numpy Array of the specified shape containing the spectrum-specified noise.
    '''
    if color.lower() == 'white':
        return get_white(shape)
    elif color.lower() == 'pink':
        return get_pink(shape)
    elif color.lower() == 'brown':
        return get_brown(shape)
    elif color.lower() == 'blue':
        return get_blue(shape)
    elif color.lower() == 'violet':
        return get_violet(shape)
    else:
        print('WARNING: Requested noise color not supported.  Returning white noise!')
        return get_white(shape)


if __name__ == '__main__':
    from numpy import square, abs, sort, argsort
    from scipy import fft
    import numpy as np
    def get_fft( data, fs):
        '''Returns the fft of the input data in a sorted and ready-to-plot structure.
        
        Tip:
            This Function can handle multiple channels in an numpy ndarray.

        Args:
            data (numpy array): Numpy array or ndarray representing the data to be filtered.
            fs (int):           Sample rate of the data.

        Returns:
            freq_values (plot X axis):  Frequency axis (Hz) corresponding to the output in power_values
            power_values (plot Y axis): numpy array or ndarray (matching the format passed) of the fft results of the input data.
        '''
        fft_power   = square( abs( fft.fft( (data.transpose() - (data.sum(axis=-1)/data.shape[-1])).transpose())))
        freq_values = fft.fftfreq( data.shape[-1], d=1/fs)
        try:
            power_sorted = fft_power[:, argsort(freq_values)]
        except IndexError:
            power_sorted = fft_power[argsort(freq_values)]
        return sort(freq_values), power_sorted

    import matplotlib.pyplot as plt

    time       = np.arange( 10000)
    noise      = get_color( time.shape, 'pink')

    freq, power = get_fft( noise, 1)
    plt.subplot(2,1,1)
    plt.plot( time, noise)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.subplot(2,1,2)
    plt.loglog( freq, power)
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.show()


    multichannel = np.array([time, time, time])
    noise        = get_pink( multichannel.shape)
    noise        = get_noise( multichannel.shape, 1.5)

    show_channel = 2
    freq, power = get_fft( noise[show_channel, :], 1)
    plt.subplot(2,1,1)
    plt.plot( time, noise[show_channel,:])
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.subplot(2,1,2)
    plt.loglog( freq, power)
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.show()