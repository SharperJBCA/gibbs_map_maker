import numpy as np
from gibbs_map_maker.gibbs_sampler import GibbsSampler
from gibbs_map_maker.matrix import MatrixA, MatrixA2 ,MatrixB2, PointingMatrix, OffsetMatrix, GradientMatrix

def test_gibbs_sampler():
    # Create an instance of MatrixA
    matrix_a = MatrixA(rows=2, cols=2)

    # Create the vector d
    test_Av = np.ones(2) 
    vector_d = matrix_a.forward(test_Av) 

    # Create an instance of the GibbsSampler
    sampler = GibbsSampler(matrices=[matrix_a], vector_d=vector_d)

    # Perform Gibbs sampling
    sampler.gibbs_sampling()

    # Get the resulting parameter vectors
    parameter_vectors = sampler.get_parameter_vectors()

    # Extract the solution
    x = parameter_vectors['matrixa']

    # Print the solution
    print("Solution: x =", x)

    # Check if the solution is close to the true solution
    true_solution = np.array([1, 1])
    assert np.allclose(x, true_solution, rtol=1e-6), "Solution does not match the true solution"

    print("Test passed!")

def test_gibbs_sampler_multiple_matrices():
    # Create instances of MatrixA and MatrixB
    matrix_a = MatrixA2(rows=6, cols=2)
    matrix_b = MatrixB2(rows=6, cols=3)

    # Create the vector d
    test_Av = np.array([3,2]) 
    test_Bv = np.array([2,3,1])
    vector_d = matrix_a.forward(test_Av) + matrix_b.forward(test_Bv)
    print(vector_d.shape)

    # Create an instance of the GibbsSampler
    sampler = GibbsSampler(matrices=[matrix_a, matrix_b], vector_d=vector_d)

    # Perform Gibbs sampling
    sampler.gibbs_sampling()

    # Get the resulting parameter vectors
    parameter_vectors = sampler.get_parameter_vectors()

    # Extract the solutions for a and b
    a = parameter_vectors['matrixa2']
    b = parameter_vectors['matrixb2']

    # Print the solutions
    print("Solution: a =", a)
    print("Solution: b =", b)
    vector_d_out = matrix_a.forward(a) + matrix_b.forward(b)
    print('Output vs input:', vector_d_out, vector_d)

    # Check if the solutions are close to the true solutions
    assert np.allclose(vector_d, vector_d_out, rtol=1e-6), "Solution for a does not match the true solution"

    print("Test passed!")

def test_gibbs_sampler_noise_and_sky():
    from astropy.io import fits
    from astropy.wcs import WCS
    import numpy as np
    from matplotlib import pyplot 

    # Read in map for signal
    hdu = fits.open('../../CBASS_PolarisedTF/ancillary_data/R1_ancil_maps/R1_iris100.fits')
    wcs = WCS(hdu[0].header)
    signal_map = hdu[0].data
    ra_min = wcs.all_pix2world(0, 0, 0)[0]
    ra_max = wcs.all_pix2world(signal_map.shape[1], 0, 0)[0]
    dec_min = wcs.all_pix2world(0, 0, 0)[1]
    dec_max = wcs.all_pix2world(0, signal_map.shape[0], 0)[1]

    # Create pointing tracks
    num_dec_steps = 100
    num_ra_steps = 100
    dec_step = (dec_max - dec_min) / num_dec_steps
    ra_step = (ra_max - ra_min) / num_ra_steps

    # Generate RA and Dec values for the scanning pattern
    ra_values = []
    dec_values = []

    # Scan in RA direction
    for dec in np.arange(dec_max, dec_min, -dec_step):
        if len(ra_values) % 2 == 0:
            ra_scan = np.arange(ra_min, ra_max, ra_step)
        else:
            ra_scan = np.arange(ra_max, ra_min, -ra_step)
        ra_values.extend(ra_scan)
        dec_values.extend([dec] * len(ra_scan))

    # Scan in Dec direction
    for ra in np.arange(ra_min, ra_max, ra_step):
        if len(dec_values) % 2 == 0:
            dec_scan = np.arange(dec_min, dec_max, dec_step)
        else:
            dec_scan = np.arange(dec_max, dec_min, -dec_step)
        ra_values.extend([ra] * len(dec_scan))
        dec_values.extend(dec_scan)

    pyplot.plot(ra_values, dec_values,'-')
    pyplot.savefig('test_track.png')
    pyplot.close()

    # Output map
    wcs_output = wcs.deepcopy()
    header = wcs_output.to_header()
    header['NAXIS1'] = 150
    header['NAXIS2'] = 150
    header['CRPIX1'] = 75.5
    header['CRPIX2'] = 75.5
    header['CDELT1'] = -0.25
    header['CDELT2'] = 0.25
    wcs_output = WCS(header)
    output_map = np.zeros((150,150))


    # Convert RA and Dec values to pixel coordinates
    pixel_coords = wcs_output.all_world2pix(ra_values, dec_values, 0)
    pixel_coords = np.array(pixel_coords).T

    # Unravel the pixel coordinates
    flat_pixels = np.ravel_multi_index((pixel_coords[:, 1].astype(int), pixel_coords[:, 0].astype(int)), dims=output_map.shape,mode='wrap')

    # Extract signal values at the scanned locations
    signal_values = signal_map[pixel_coords[:, 0].astype(int), pixel_coords[:, 1].astype(int)]
    signal_values[np.isnan(signal_values)] = 0

    # Bin the signal values into the output map
    test_output = np.histogram(flat_pixels, bins=np.arange(output_map.size + 1), weights=signal_values)[0]
    test_output_bot = np.histogram(flat_pixels, bins=np.arange(output_map.size + 1))[0]
    test_output_map = test_output / test_output_bot
    

    # Add noise to the signal values
    noise_std = 0.1  # Adjust the noise level as needed
    noise = np.random.normal(0, noise_std, len(signal_values))
    observed_values = signal_values + noise

    # Create MatrixA and vector_d
    matrix_a = PointingMatrix(rows=len(observed_values), cols=output_map.size, data={'pixels': flat_pixels, 'npix': output_map.size})
    vector_d = observed_values

    # Create an instance of the GibbsSampler
    sampler = GibbsSampler(matrices=[matrix_a], vector_d=vector_d)

    # Perform Gibbs sampling
    sampler.gibbs_sampling()

    # Get the resulting parameter vectors
    parameter_vectors = sampler.get_parameter_vectors()

    # Extract the solution
    reconstructed_signal = parameter_vectors['pointingmatrix']

    pyplot.subplot(131)
    pyplot.imshow(reconstructed_signal.reshape(output_map.shape), origin='lower', cmap='inferno')
    pyplot.colorbar()
    pyplot.subplot(132)
    pyplot.imshow(test_output_map.reshape(output_map.shape), origin='lower', cmap='inferno')
    pyplot.colorbar()
    pyplot.subplot(133)
    pyplot.imshow(reconstructed_signal.reshape(output_map.shape)-test_output_map.reshape(output_map.shape), origin='lower', cmap='inferno')
    pyplot.colorbar()
    pyplot.savefig('test.png')


    # Compare the reconstructed signal with the original signal
    # Perform any necessary analysis or visualization

    print("Test completed!")


def test_gibbs_sampler_correlated_noise_and_sky():
    from astropy.io import fits
    from astropy.wcs import WCS
    import numpy as np
    from matplotlib import pyplot 

    # Read in map for signal
    hdu = fits.open('../../CBASS_PolarisedTF/ancillary_data/R1_ancil_maps/R1_iris100.fits')
    wcs = WCS(hdu[0].header)
    signal_map = hdu[0].data
    ra_min = wcs.all_pix2world(0, 0, 0)[0]
    ra_max = wcs.all_pix2world(signal_map.shape[1], 0, 0)[0]
    dec_min = wcs.all_pix2world(0, 0, 0)[1]
    dec_max = wcs.all_pix2world(0, signal_map.shape[0], 0)[1]

    # Create pointing tracks
    num_dec_steps = 100
    num_ra_steps = 100
    dec_step = (dec_max - dec_min) / num_dec_steps
    ra_step = (ra_max - ra_min) / num_ra_steps

    # Generate RA and Dec values for the scanning pattern
    ra_values = []
    dec_values = []

    # Scan in RA direction
    for dec in np.arange(dec_max, dec_min, -dec_step):
        if len(ra_values) % 2 == 0:
            ra_scan = np.arange(ra_min, ra_max, ra_step)
        else:
            ra_scan = np.arange(ra_max, ra_min, -ra_step)
        ra_values.extend(ra_scan)
        dec_values.extend([dec] * len(ra_scan))

    # Scan in Dec direction
    for ra in np.arange(ra_min, ra_max, ra_step):
        if len(dec_values) % 2 == 0:
            dec_scan = np.arange(dec_min, dec_max, dec_step)
        else:
            dec_scan = np.arange(dec_max, dec_min, -dec_step)
        ra_values.extend([ra] * len(dec_scan))
        dec_values.extend(dec_scan)

    pyplot.plot(ra_values, dec_values,'-')
    pyplot.savefig('test_track.png')
    pyplot.close()

    # Output map
    wcs_output = wcs.deepcopy()
    header = wcs_output.to_header()
    header['NAXIS1'] = 150
    header['NAXIS2'] = 150
    header['CRPIX1'] = 75.5
    header['CRPIX2'] = 75.5
    header['CDELT1'] = -0.25
    header['CDELT2'] = 0.25
    wcs_output = WCS(header)
    output_map = np.zeros((150,150))


    # Convert RA and Dec values to pixel coordinates
    pixel_coords = wcs_output.all_world2pix(ra_values, dec_values, 0)
    pixel_coords = np.array(pixel_coords).T

    # Unravel the pixel coordinates
    flat_pixels = np.ravel_multi_index((pixel_coords[:, 1].astype(int), pixel_coords[:, 0].astype(int)), dims=output_map.shape,mode='wrap')

    # Extract signal values at the scanned locations
    signal_values = signal_map[pixel_coords[:, 0].astype(int), pixel_coords[:, 1].astype(int)]
    signal_values[np.isnan(signal_values)] = 0

    # Bin the signal values into the output map
    test_output = np.histogram(flat_pixels, bins=np.arange(output_map.size + 1), weights=signal_values)[0]
    test_output_bot = np.histogram(flat_pixels, bins=np.arange(output_map.size + 1))[0]
    test_output_map = test_output / test_output_bot
    

    # Add noise to the signal values
    noise_std = 0.1  # Adjust the noise level as needed
    noise = np.random.normal(0, noise_std, len(signal_values))
    ft_noise = np.fft.fft(noise)
    ft_modes = np.fft.fftfreq(len(signal_values))
    ft_modes[0] = ft_modes[1]
    power_spec = (1 + 0.01*np.abs(ft_modes)**-2)
    noise = np.fft.ifft(ft_noise * power_spec**0.5).real
    observed_values = signal_values + noise

    # Bin the naive values into the output map
    test_naive = np.histogram(flat_pixels, bins=np.arange(output_map.size + 1), weights=observed_values)[0]
    test_naive_bot = np.histogram(flat_pixels, bins=np.arange(output_map.size + 1))[0]
    test_naive_map = test_naive / test_naive_bot

    # Create MatrixA and vector_d
    matrix_a = PointingMatrix(rows=len(observed_values), cols=output_map.size, data={'pixels': flat_pixels, 'npix': output_map.size})

    offset_size = 31 
    flat_offsets = np.arange(observed_values.size)//offset_size 
    noff = np.unique(flat_offsets).size
    matrix_b = OffsetMatrix(rows=len(observed_values), cols=noff, data={'offsets': flat_offsets, 'noff': noff})
    vector_d = observed_values

    # Create an instance of the GibbsSampler
    sampler = GibbsSampler(matrices=[matrix_a,matrix_b], vector_d=vector_d)

    # Perform Gibbs sampling
    sampler.gibbs_sampling()

    # Get the resulting parameter vectors
    parameter_vectors = sampler.get_parameter_vectors()

    # Extract the solution
    reconstructed_signal = parameter_vectors['pointingmatrix']
    recovered_noise = parameter_vectors['offsetmatrix']

    reconstructed_signal[reconstructed_signal==0] = np.nan
    pyplot.subplot(131)
    pyplot.imshow(reconstructed_signal.reshape(output_map.shape), origin='lower', cmap='inferno')
    pyplot.colorbar()
    pyplot.subplot(132)
    pyplot.imshow(test_output_map.reshape(output_map.shape), origin='lower', cmap='inferno')
    pyplot.colorbar()
    pyplot.subplot(133)
    pyplot.imshow(test_naive_map.reshape(output_map.shape), origin='lower', cmap='inferno')
    pyplot.colorbar()
    pyplot.savefig('test.png')
    pyplot.close()

    pyplot.plot(noise)
    pyplot.plot(recovered_noise[flat_offsets])
    pyplot.savefig('test_noise.png')
    pyplot.close()

    # Compare the reconstructed signal with the original signal
    # Perform any necessary analysis or visualization

    print("Test completed!")


def test_gibbs_sampler_correlated_noise_and_sky_and_systematic():
    from astropy.io import fits
    from astropy.wcs import WCS
    import numpy as np
    from matplotlib import pyplot 
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    pyplot.rcParams['font.size'] = 10
    pyplot.rcParams['axes.labelsize'] = 10
    pyplot.rcParams['xtick.labelsize'] = 10
    pyplot.rcParams['ytick.labelsize'] = 10

    # Read in map for signal
    hdu = fits.open('../../CBASS_PolarisedTF/ancillary_data/R1_ancil_maps/R1_iris100.fits')
    wcs = WCS(hdu[0].header)
    signal_map = hdu[0].data
    ra_min = wcs.all_pix2world(0, 0, 0)[0]
    ra_max = wcs.all_pix2world(signal_map.shape[1], 0, 0)[0]
    dec_min = wcs.all_pix2world(0, 0, 0)[1]
    dec_max = wcs.all_pix2world(0, signal_map.shape[0], 0)[1]

    # Create pointing tracks
    num_dec_steps = 100
    num_ra_steps = 100
    dec_step = (dec_max - dec_min) / num_dec_steps
    ra_step = (ra_max - ra_min) / num_ra_steps

    # Generate RA and Dec values for the scanning pattern
    ra_values = []
    dec_values = []

    # Scan in RA direction
    for dec in np.arange(dec_max, dec_min, -dec_step):
        if len(ra_values) % 2 == 0:
            ra_scan = np.arange(ra_min, ra_max, ra_step)
        else:
            ra_scan = np.arange(ra_max, ra_min, -ra_step)
        ra_values.extend(ra_scan)
        dec_values.extend([dec] * len(ra_scan))

    # Scan in Dec direction
    for ra in np.arange(ra_min, ra_max, ra_step):
        if len(dec_values) % 2 == 0:
            dec_scan = np.arange(dec_min, dec_max, dec_step)
        else:
            dec_scan = np.arange(dec_max, dec_min, -dec_step)
        ra_values.extend([ra] * len(dec_scan))
        dec_values.extend(dec_scan)


    pyplot.plot(ra_values, dec_values,'-')
    pyplot.title('Scanning Pattern')
    pyplot.xlabel('RA')
    pyplot.ylabel('Dec')
    pyplot.savefig('test_track.png')
    pyplot.close()

    # Output map
    wcs_output = wcs.deepcopy()
    header = wcs_output.to_header()
    header['NAXIS1'] = 150
    header['NAXIS2'] = 150
    header['CRPIX1'] = 75.5
    header['CRPIX2'] = 75.5
    header['CDELT1'] = -0.25
    header['CDELT2'] = 0.25
    wcs_output = WCS(header)
    output_map = np.zeros((150,150))


    # Convert RA and Dec values to pixel coordinates
    pixel_coords = wcs_output.all_world2pix(ra_values, dec_values, 0)
    pixel_coords = np.array(pixel_coords).T

    # Unravel the pixel coordinates
    flat_pixels = np.ravel_multi_index((pixel_coords[:, 1].astype(int), pixel_coords[:, 0].astype(int)), dims=output_map.shape,mode='wrap')

    # Extract signal values at the scanned locations
    signal_values = signal_map[pixel_coords[:, 0].astype(int), pixel_coords[:, 1].astype(int)]
    signal_values[np.isnan(signal_values)] = 0

    # Bin the signal values into the output map
    test_output = np.histogram(flat_pixels, bins=np.arange(output_map.size + 1), weights=signal_values)[0]
    test_output_bot = np.histogram(flat_pixels, bins=np.arange(output_map.size + 1))[0]
    test_output_map = test_output / test_output_bot
    

    # Add noise to the signal values
    noise_std = 0.1  # Adjust the noise level as needed
    noise = np.random.normal(0, noise_std, len(signal_values))
    ft_noise = np.fft.fft(noise)
    ft_modes = np.fft.fftfreq(len(signal_values))
    ft_modes[0] = ft_modes[1]
    power_spec = (1 + 0.0004*np.abs(ft_modes)**-2)
    noise = np.fft.ifft(ft_noise * power_spec**0.5).real



    # Create MatrixA and vector_d
    matrix_a = PointingMatrix(rows=len(signal_values), cols=output_map.size, data={'pixels': flat_pixels, 'npix': output_map.size})

    offset_size = 31 
    flat_offsets = np.arange(signal_values.size)//offset_size 
    noff = np.unique(flat_offsets).size
    matrix_b = OffsetMatrix(rows=len(signal_values), cols=noff, data={'offsets': flat_offsets, 'noff': noff})

    min_ra = np.min(ra_values)
    max_ra = np.max(ra_values)
    range_ra = max_ra - min_ra
    mid_ra = (max_ra + min_ra) / 2
    norm_ra_values = (ra_values - mid_ra) / range_ra
    matrix_c = GradientMatrix(rows=len(signal_values), cols=1, data={'coordinate': norm_ra_values})
    gradient_amplitude = 1.
    # Create a systematic gradient that is correlated with RA
    
    observed_values = signal_values + noise + norm_ra_values*gradient_amplitude

    # Bin the naive values into the output map
    test_naive = np.histogram(flat_pixels, bins=np.arange(output_map.size + 1), weights=observed_values)[0]
    test_naive_bot = np.histogram(flat_pixels, bins=np.arange(output_map.size + 1))[0]
    test_naive_map = test_naive / test_naive_bot

    vector_d = observed_values

    # Create an instance of the GibbsSampler
    sampler = GibbsSampler(matrices=[matrix_a,matrix_b,matrix_c], vector_d=vector_d,initial_values={'gradientmatrix':np.array([0.])},max_iterations=100)

    # Perform Gibbs sampling
    sampler.gibbs_sampling()

    # Get the resulting parameter vectors
    parameter_vectors = sampler.get_parameter_vectors()

    # Extract the solution
    reconstructed_signal = parameter_vectors['pointingmatrix']
    recovered_noise = parameter_vectors['offsetmatrix']
    recovered_gradient = parameter_vectors['gradientmatrix']

    pyplot.plot(np.array(sampler.parameter_chains['gradientmatrix']).flatten())
    pyplot.axhline(gradient_amplitude,linestyle='--',color='r',label='Truth')
    pyplot.legend()
    pyplot.xlabel('Gibbs Iteration')
    pyplot.ylabel('RA Gradient')
    pyplot.title('Gradient Gibbs Chain')
    pyplot.savefig('test_gradient_chain.png')
    pyplot.close()

    print(recovered_gradient)
    pyplot.plot(norm_ra_values*recovered_gradient)
    pyplot.savefig('test_gradient.png')
    pyplot.close()

    reconstructed_signal[reconstructed_signal==0] = np.nan

    figure = pyplot.figure()
    axes = pyplot.subplot(131)
    pyplot.imshow(reconstructed_signal.reshape(output_map.shape), origin='lower', cmap='inferno')
    pyplot.title('Destriped Map')
    divider = make_axes_locatable(axes)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=pyplot.Axes)
    if len(axes.images) > 0:
        cbar = figure.colorbar(axes.images[0], ax=axes, cax=cax)
    axes = pyplot.subplot(132)
    pyplot.imshow(test_output_map.reshape(output_map.shape), origin='lower', cmap='inferno')
    pyplot.title('Input Map (signal)')
    divider = make_axes_locatable(axes)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=pyplot.Axes)
    if len(axes.images) > 0:
        cbar = figure.colorbar(axes.images[0], ax=axes, cax=cax)
    axes = pyplot.subplot(133)
    pyplot.imshow(test_naive_map.reshape(output_map.shape), origin='lower', cmap='inferno')
    pyplot.title('Naive Map (signal + noise)')
    divider = make_axes_locatable(axes)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=pyplot.Axes)
    if len(axes.images) > 0:
        cbar = figure.colorbar(axes.images[0], ax=axes, cax=cax)
    pyplot.subplots_adjust(wspace=0.5)
    pyplot.savefig('test.png')
    pyplot.close()

    pyplot.plot(noise,label='input noise')
    pyplot.plot(recovered_noise[flat_offsets],label='fitted noise')
    pyplot.legend()
    pyplot.xlabel('Sample')
    pyplot.savefig('test_noise.png')
    pyplot.close()


test_gibbs_sampler_correlated_noise_and_sky_and_systematic()
#test_gibbs_sampler()
#test_gibbs_sampler_multiple_matrices() 