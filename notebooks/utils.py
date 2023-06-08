import glob
import re

def build_coordinate_mapping(source_image, target_image, h5_forward, h5_inverse, output_dir='./', file_name=None,
                             verbose=False, save_data=True):
    from nipype.interfaces.ants import ApplyTransforms
    import nibabel as nb
    from nighres.io import load_volume, save_volume
    from nighres.utils import _output_dir_4saving, _fname_4saving, _check_topology_lut_dir

    X=0
    Y=1
    Z=2
    T=3
    
    # load
    if verbose:
        print('Loading source & target...')
    source = load_volume(source_image)
    src_affine = source.affine
    src_header = source.header
    nsx = source.header.get_data_shape()[X]
    nsy = source.header.get_data_shape()[Y]
    nsz = source.header.get_data_shape()[Z]
    rsx = source.header.get_zooms()[X]
    rsy = source.header.get_zooms()[Y]
    rsz = source.header.get_zooms()[Z]

    target = load_volume(target_image)
    trg_affine = target.affine
    trg_header = target.header
    ntx = target.header.get_data_shape()[X]
    nty = target.header.get_data_shape()[Y]
    ntz = target.header.get_data_shape()[Z]
    rtx = target.header.get_zooms()[X]
    rty = target.header.get_zooms()[Y]
    rtz = target.header.get_zooms()[Z]
    
    if verbose:
        print('Building coordinate mappings...')
    # build coordinate mappings
    src_coord = np.zeros((nsx,nsy,nsz,3))
    trg_coord = np.zeros((ntx,nty,ntz,3))
    for x in range(nsx):
        for y in range(nsy):
            for z in range(nsz):
                src_coord[x,y,z,X] = x
                src_coord[x,y,z,Y] = y
                src_coord[x,y,z,Z] = z
    src_map = nb.Nifti1Image(src_coord, source.affine, source.header)
    src_map_file = os.path.join(output_dir, _fname_4saving(file_name=file_name,
                                                           rootfile=source_image,
                                                           suffix='tmp_srccoord'))
    save_volume(src_map_file, src_map)
    for x in range(ntx):
        for y in range(nty):
            for z in range(ntz):
                trg_coord[x,y,z,X] = x
                trg_coord[x,y,z,Y] = y
                trg_coord[x,y,z,Z] = z
    trg_map = nb.Nifti1Image(trg_coord, target.affine, target.header)
    trg_map_file = os.path.join(output_dir, _fname_4saving(file_name=file_name,
                                                           rootfile=source_image,
                                                           suffix='tmp_trgcoord'))
    save_volume(trg_map_file, trg_map)
    
#     if verbose:
#         print('Applying transforms to source...')
#     at = ApplyTransforms()
#     at.inputs.dimension = 2
#     at.inputs.input_image = source.get_filename()
#     at.inputs.reference_image = target.get_filename()
#     at.inputs.interpolation = 'NearestNeighbor'
#     at.inputs.transforms = h5_forward
# #    at.inputs.invert_transform_flags = result.outputs.forward_invert_flags
#     print(at.cmdline)
#     transformed = at.run()

    if verbose:
        print('Applying transforms to forward...')
    # Create coordinate mappings
    src_at = ApplyTransforms()
    src_at.inputs.dimension = 3
    src_at.inputs.input_image_type = 3
    src_at.inputs.input_image = src_map.get_filename()
    src_at.inputs.reference_image = target.get_filename()
    src_at.inputs.interpolation = 'Linear'
    src_at.inputs.transforms = h5_forward
#    src_at.inputs.invert_transform_flags = result.outputs.forward_invert_flags
    mapping = src_at.run()

    if verbose:
        print('Applying transforms to inverse...')
    trg_at = ApplyTransforms()
    trg_at.inputs.dimension = 3
    trg_at.inputs.input_image_type = 3
    trg_at.inputs.input_image = trg_map.get_filename()
    trg_at.inputs.reference_image = source.get_filename()
    trg_at.inputs.interpolation = 'Linear'
    trg_at.inputs.transforms = h5_inverse
#    trg_at.inputs.invert_transform_flags = result.outputs.reverse_invert_flags
    inverse = trg_at.run()
    
    # save - already done?
    if verbose:
        print('Creating niftis...')
    mapping_img = nb.Nifti1Image(nb.load(mapping.outputs.output_image).get_data(),
                                    target.affine, target.header)
    inverse_img = nb.Nifti1Image(nb.load(inverse.outputs.output_image).get_data(),
                                    source.affine, source.header)

    outputs = {'mapping': mapping_img,
               'inverse': inverse_img}

    if verbose:
        print('Clean-up & save...')
    # clean-up intermediate files
    os.remove(src_map_file)
    os.remove(trg_map_file)

    os.remove(mapping.outputs.output_image)
    os.remove(inverse.outputs.output_image)

    if save_data:
        mapping_file = os.path.join(output_dir,
                                    _fname_4saving(file_name=file_name,
                                               rootfile=source_image,
                                               suffix='ants-map'))

        inverse_mapping_file = os.path.join(output_dir,
                                            _fname_4saving(file_name=file_name,
                                                    rootfile=source_image,
                                                    suffix='ants-invmap'))
#         save_volume(transformed_source_file, transformed_img)
        save_volume(mapping_file, mapping_img)
        save_volume(inverse_mapping_file, inverse_img)
        
    return outputs

def load_atlas(resolution='1p6mm'):
    ### Rois in MNI09c-space
    mask_dir='./masks/final_masks_mni09c_' + resolution
    fns = glob.glob(mask_dir + '/space-*')
    fns.sort()
    names = [re.match('.*space-(?P<space>[a-zA-Z0-9]+)_label-(?P<label>[a-zA-Z0-9]+)_probseg.nii.gz', fn).groupdict()['label'] for fn in fns]
    roi_dict = dict(zip(names, fns))
    
    from nilearn import image
    combined = image.concat_imgs(roi_dict.values())
    
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self
            
    roi_atlas = AttrDict({'maps': combined,
                          'labels': roi_dict.keys()})
    
    return roi_atlas


# functions for combining echos, based on the tedana workflow
def combine_tedana(tes, data, combmodes=('t2s', 'ste'), mask=None, overwrite=True):
    """ Function based on tedana main workflow """
    from tedana import utils, model, io, decay, combine
    from scipy import stats
    import numpy as np
    import os
    
    # ensure tes are in appropriate format
    tes = [float(te) for te in tes]
    n_echos = len(tes)

    # coerce data to samples x echos x time array
    if isinstance(data, str):
        data = [data]

    catd, ref_img = io.load_data(data, n_echos=n_echos)
    n_samp, n_echos, n_vols = catd.shape
    
    mask, masksum = utils.make_adaptive_mask(catd, mask=mask, minimum=False, getsum=True)

    # check if the t2s-map is already created first
    base_name = data[0].replace('_echo-1', '').replace('desc-preproc-hp', 'desc-preproc-hp-%s').replace('.nii', '').replace('.gz', '')    
    
    if not os.path.exists(base_name %'t2sv' + '.nii.gz') or overwrite:
        t2s, s0, t2ss, s0s, t2sG, s0G = decay.fit_decay(catd, tes, mask, masksum)
        # set a hard cap for the T2* map
        # anything that is 10x higher than the 99.5 %ile will be reset to 99.5 %ile
        cap_t2s = stats.scoreatpercentile(t2s.flatten(), 99.5,
                                          interpolation_method='lower')
        t2s[t2s > cap_t2s * 10] = cap_t2s
        
        # save
        io.filewrite(t2s, base_name %'t2sv' + '.nii', ref_img, gzip=True)
        io.filewrite(s0, base_name %'s0v' + '.nii', ref_img, gzip=True)
        io.filewrite(t2ss, base_name %'t2ss' + '.nii', ref_img, gzip=True)
        io.filewrite(s0s, base_name %'s0vs' + '.nii', ref_img, gzip=True)
        io.filewrite(t2sG, base_name %'t2svG' + '.nii', ref_img, gzip=True)
        io.filewrite(s0G, base_name %'s0vG' + '.nii', ref_img, gzip=True)
        
    else:
        t2sG = utils.load_image(base_name %'t2svG' + '.nii.gz')
        t2s = utils.load_image(base_name %'t2sv' + '.nii.gz')
        
    # optimally combine data
    if 't2s' in combmodes:
        print('Combining echos using optcomb...', end='')
        ext = 'optcomb'
        data_oc = combine.make_optcom(catd, tes, mask, t2s=t2sG, combmode='t2s')
        # make sure to set all nan-values/inf to 0
        data_oc[np.isinf(data_oc)] = 0
        data_oc[np.isnan(data_oc)] = 0
        print('Done, writing results...')
        io.filewrite(data_oc, base_name %ext + '.nii', ref_img, gzip=True)
    if 'ste' in combmodes:
        print('Combining echos using optcomb...', end='')
        ext = 'PAID'
        data_oc = combine.make_optcom(catd, tes, mask, t2s=t2sG, combmode='ste')
        # make sure to set all nan-values/inf to 0
        data_oc[np.isinf(data_oc)] = 0
        data_oc[np.isnan(data_oc)] = 0
        io.filewrite(data_oc, base_name %ext + '.nii', ref_img, gzip=True)
        print('Done, writing results...')
        
    return 0

# create tsnr
import nibabel as nib
import numpy as np
def tsnr_img(hdr):
    if isinstance(hdr, str):
        hdr = nib.load(hdr)
        
    dat = hdr.get_data()
    mn = np.mean(dat, 3)
    sd = np.std(dat, 3)
    
    img = nib.Nifti1Image(mn/sd, hdr.affine)
    return img



###### Supplementary plotting functions
def get_color(mask):
    if 'STN' in mask:
        return 'lightblue'
    if 'STR' in mask:
        return 'blue'
    if 'PreSMA' in mask:
        return 'darkgreen'
    if 'ACC' in mask:
        return 'green'
    if 'M1' in mask:
        return 'pink'
    if 'GPi' in mask:
        return 'lightgreen'
    if 'GPe' in mask:
        return 'green'
    if 'IFG' in mask:
        return 'white'
    if 'VTA' in mask:
        return 'lightgreen'
    if 'SN' in mask:
        return 'pink'

def get_roi_dict():
    # make dict of masks & filenames in 09c-space, get colors
    fns = glob.glob('./masks/final_masks_mni09c_1mm/space*')
    fns.sort()
    names = [re.match('.*space-(?P<space>[a-zA-Z0-9]+)_label-(?P<label>[a-zA-Z0-9]+)_probseg.nii.gz', fn).groupdict()['label'] for fn in fns]
    roi_dict = dict(zip(names, fns))
    for mask, fn in roi_dict.items():
        roi_dict[mask] = {}
        roi_dict[mask]['fn'] = fn
        roi_dict[mask]['color'] = get_color(mask)
        roi_dict[mask]['threshold'] = 0.3
    return roi_dict

def get_prop_limits(props, current_limits):
    extent = current_limits[1]-current_limits[0]
    x0 = current_limits[0] + extent*props[0]
    x1 = current_limits[0] + extent*props[1]
    return (x0, x1)

def add_contours(disp, roi, color='white', linewidth=2, thr=0.3, **kwargs):
    from nilearn._utils.extmath import fast_abs_percentile
    from nilearn._utils.param_validation import check_threshold
    if not isinstance(roi, nib.Nifti1Image):
        map_img = nib.load(roi)
    else:
        map_img = roi
    data = map_img.get_data()
    
    # manually threhsold image
    data[data < thr] = 0
    
    # then determine the plotting threshold - this is a different value, required for plotting reasons,
    # and finds the percentile of the data that corresponds to the threshold
#     thr = check_threshold(thr, data,
#                           percentile_func=fast_abs_percentile,
#                           name='threshold')
    
    # Get rid of background values in all cases
    thr = max(thr, 1e-6)
    disp.add_contours(nib.Nifti1Image(data, map_img.affine), levels=[thr], linewidths=linewidth, colors=[color], **kwargs)

from nilearn import plotting
def draw_custom_colorbar(colorbar_ax, vmin=3, vmax=6, truncation_limits=(0,6), offset=4., nb_ticks=4, flip=True,
                         format="%d", cmap=plotting.cm.cold_hot):
    from matplotlib.colorbar import ColorbarBase
    from matplotlib import colors
    our_cmap = cmap
    if flip:
        truncation_limits = [truncation_limits[1], truncation_limits[0]]
    ticks = np.linspace(truncation_limits[0], truncation_limits[1], nb_ticks)
    bounds = np.linspace(truncation_limits[0], truncation_limits[1], our_cmap.N)
    norm = colors.Normalize(vmin=-vmax, vmax=vmax)
    
    # some colormap hacking
    cmaplist = [our_cmap(i) for i in range(our_cmap.N)]
    istart = int(norm(-offset, clip=True) * (our_cmap.N - 1))
    istop = int(norm(offset, clip=True) * (our_cmap.N - 1))
    for i in range(istart, istop):
        cmaplist[i] = (0.5, 0.5, 0.5, 1.)  # just an average gray color
    our_cmap = our_cmap.from_list('Custom cmap', cmaplist, our_cmap.N)

    ColorbarBase(colorbar_ax, ticks=ticks, norm=norm,
                 orientation='vertical', cmap=our_cmap, boundaries=bounds,
                 spacing='proportional', format=format)
    
    if flip:
        colorbar_ax.invert_yaxis()
        colorbar_ax.yaxis.tick_right()
    else:
        colorbar_ax.yaxis.tick_left()

#     tick_color = 'w'
#     for tick in colorbar_ax.yaxis.get_ticklabels():
#         tick.set_color(tick_color)
#     colorbar_ax.yaxis.set_tick_params(width=0)
    return colorbar_ax

    

########## Plotting functions
from nilearn import plotting
from matplotlib import gridspec
import matplotlib.pyplot as plt

def plot_spm(zmaps, roi_dict, bg_img=None, z_threshold=0, f=None, axes=None,
             # brain_mask='../Templates/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c.nii.gz',
             roi_to_plot=('PreSMA', 'M1', 'ACC', 'rIFG', 'STR', 'GPe', 'GPi', 'STN'),
             cut_coords=[None, None, None, None, None, None, None, None],
             contrasts=('failed_stop - go_trial',
                        'successful_stop - go_trial',
                        'failed_stop - successful_stop'),
             plot_columns=(0, 1, 3, 4, 6, 7),
             empty_plots=False, skip_all_but_last=False,
             **kwargs):
    
    if f is None:
        gridspec = dict(hspace=0.0, wspace=0.0, width_ratios=[1, 1, 0.05, 1, 1, .05, 1, 1, .1])
        f, axes = plt.subplots(len(roi_to_plot), len(zmaps)+3, gridspec_kw=gridspec)  # add 3 columns: 2 interspace, 1 on the right for the colorbar
 
    if empty_plots:
        f.set_size_inches(len(zmaps)*4, len(roi_to_plot)*4)
        return f, axes
    
    all_cut_coords = []
    all_disps = []
    for row_n, roi in enumerate(roi_to_plot):
        # for debugging
        if skip_all_but_last:
            if row_n < (len(roi_to_plot)-1):
                continue
        
        # get cut coordinates based on 1 hemisphere (if applicable)
        if roi in ['STR', 'STN', 'PreSMA', 'GPe', 'GPi']:
            roi_map = roi_dict['l' + roi]
        else:
            roi_map = roi_dict[roi]
#        roi_map = make_conjunction_mask(roi_map['fn'], brain_mask)
        if roi == 'rIFG':
            ## saggital
            if cut_coords[row_n] is None:
                this_cut_coords = plotting.find_xyz_cut_coords(roi_map['fn'])[0:1]
            else:
                this_cut_coords = cut_coords[row_n]
            display_mode='x'
            plot_rois = ['rIFG']#, 'M1', 'rPreSMA']
        elif roi == 'STR':
            ## axial view
            if cut_coords[row_n] is None:
                this_cut_coords = plotting.find_xyz_cut_coords(roi_map['fn'])[2:3]
            else:
                this_cut_coords = cut_coords[row_n]

            display_mode='z'
            plot_rois = ['rIFG', 'M1',
                         'lSTR', 'lGPe', 'lGPi', 'lSTN', 'lVTA', 'lSN',
                         'rSTR', 'rGPe', 'rGPi', 'rSTN', 'rVTA', 'rSN']
        elif roi == 'STN':
            ## plot coronal view
            if cut_coords[row_n] is None:
                this_cut_coords = plotting.find_xyz_cut_coords(roi_map['fn'])[1:2]
            else:
                this_cut_coords = cut_coords[row_n]

            display_mode='y'
            plot_rois = ['rIFG', 'M1',
                         'lSTR', 'lGPe', 'lGPi', 'lSTN', 'lVTA', 'lSN',
                         'rSTR', 'rGPe', 'rGPi', 'rSTN', 'rVTA', 'rSN']

        all_cut_coords.append({display_mode: this_cut_coords[0]})
        
        # loop over contrasts for columns
        for col_n, map_n in zip(plot_columns, np.arange(len(zmaps))):
            zmap = zmaps[map_n]
            if skip_all_but_last:
                if col_n < (len(zmaps)-1):
                    continue
            
            if row_n == (len(roi_to_plot)-1) and col_n == (len(zmaps)-1):
                # plot colobar in the last plot
                cbar = False
            else:
                cbar = False
            
#             # do not plot in column 2 or 5
#             plot_col = col_n
#             if col_n > 1:
#                 plot_col = col_n + 1
#             if col_n > 3:
#                 plot_col = col_n + 2
                
            if isinstance(z_threshold, list):
                this_threshold = z_threshold[map_n]
            else:
                this_threshold = z_threshold
            ax = axes[row_n, col_n]
            
#             print(cbar)
            disp = plotting.plot_stat_map(zmap, bg_img=bg_img, 
                                          threshold=this_threshold, cut_coords=this_cut_coords,
                                          display_mode=display_mode, axes=ax, colorbar=cbar, **kwargs)
        
            # just plot *all* contours, always
            for roi_ in plot_rois:
                roi_map = roi_dict[roi_]
#             for roi_, roi_map in roi_dict.items():
#                 print(roi_map)
                add_contours(disp, roi=roi_map['fn'], thr=roi_map['threshold'], color=roi_map['color'])

            # determine limits (xlim/ylim) based on first column, and apply to all others
            this_key = list([x for x in disp.axes.keys()])[0]
            # Determine new xlim/ylim based on first column
            if col_n == plot_columns[0]:
                # extract old/current limits
                cur_xlim = disp.axes[this_key].ax.get_xlim()
                cur_ylim = disp.axes[this_key].ax.get_ylim()
                if display_mode == 'x':
                    new_xlim = get_prop_limits([0, 1], cur_xlim)
                    new_ylim = get_prop_limits([0, 1], cur_ylim)
                elif display_mode == 'z' and 'STN' in roi:            
                    new_xlim = get_prop_limits([.25, .75], cur_xlim)
                    new_ylim = get_prop_limits([.40, .90], cur_ylim)
                elif display_mode == 'z' and 'STR' in roi:
                    new_xlim = get_prop_limits([0, 1], cur_xlim)
                    new_ylim = get_prop_limits([0.3, 1], cur_ylim)
                elif display_mode == 'y':
                    new_xlim = get_prop_limits([.26, .74], cur_xlim)
                    new_ylim = get_prop_limits([.25, .75], cur_ylim)

            # Change axes limits
            disp.axes[this_key].ax.set_xlim(new_xlim[0], new_xlim[1])
            disp.axes[this_key].ax.set_ylim(new_ylim[0], new_ylim[1])
            
            all_disps.append(disp)

#             # set new xlimits if necessary (ie zoom for STN view)
#             if 'STN' in roi and display_mode == 'z':
#                 this_key = [x for x in disp.axes.keys()]
#                 this_key = this_key[0]
#                 cur_xlim = disp.axes[this_key].ax.get_xlim()
#                 cur_ylim = disp.axes[this_key].ax.get_ylim()
#                 new_xlim = get_prop_limits([.25, .75], cur_xlim)
#                 new_ylim = get_prop_limits([.40, .90], cur_ylim)
#                 disp.axes[this_key].ax.set_xlim(new_xlim[0], new_xlim[1])
#                 disp.axes[this_key].ax.set_ylim(new_ylim[0], new_ylim[1])
#             elif 'STN' in roi and display_mode == 'y':
#                 this_key = [x for x in disp.axes.keys()]
#                 this_key = this_key[0]
#                 cur_xlim = disp.axes[this_key].ax.get_xlim()
#                 cur_ylim = disp.axes[this_key].ax.get_ylim()
#                 new_xlim = get_prop_limits([.25, .75], cur_xlim)
#                 new_ylim = get_prop_limits([.25, .75], cur_ylim)
#                 disp.axes[this_key].ax.set_xlim(new_xlim[0], new_xlim[1])
#                 disp.axes[this_key].ax.set_ylim(new_ylim[0], new_ylim[1])
#             elif 'STR' in roi and display_mode == 'z':
#                 this_key = [x for x in disp.axes.keys()]
#                 this_key = this_key[0]
#                 cur_xlim = disp.axes[this_key].ax.get_xlim()
#                 cur_ylim = disp.axes[this_key].ax.get_ylim()
#                 new_xlim = get_prop_limits([0, 1], cur_xlim)
#                 new_ylim = get_prop_limits([.3, 1], cur_ylim)
#                 disp.axes[this_key].ax.set_xlim(new_xlim[0], new_xlim[1])
#                 disp.axes[this_key].ax.set_ylim(new_ylim[0], new_ylim[1])
                
#             all_disps.append(disp)
    
    # add labels
    if not skip_all_but_last:
        for row_n, ax in enumerate(axes[:,0]):
            cc = all_cut_coords[row_n]
            disp_mode = [x for x in cc.keys()][0]
            coord = cc[disp_mode]
            ax.annotate('%s = %d' %(disp_mode, int(coord)), 
                        xy=(0, 0.5), 
                        xytext=(-ax.yaxis.labelpad - 0.5, 0),
                        xycoords=ax.yaxis.label, 
                        textcoords='offset points', rotation=90, 
                        ha='right', va='center')

    f.set_size_inches(len(zmaps)*4, len(roi_to_plot)*4)
    
    return f, axes, all_disps



def plot_3x6(zmaps, thresholds, roi_dict=get_roi_dict(),
             titles=('Single echo', 'Multi echo (OC)', 'Single echo', 'Multi echo (OC)', 'Single echo', 'Multi echo (OC)'),
             contrast_names=('Contrast 1', 'Contrast 2', 'Contrast 3'),
             vmax=6, colorbars=((3, 6), (3, 6)),
             colorbar_title='z-values',
             **cb_kwargs):
    gridspec_kws = dict(hspace=0.0, wspace=0.0, 
                    width_ratios=[1, 1, 0.05, 1, 1, .05, 1, 1, .15, .1, .1])
    gs = gridspec.GridSpec(3, len(zmaps)+5, **gridspec_kws)
    f, axes = plt.subplots(3, len(zmaps)+5, gridspec_kw=gridspec_kws)
    # add 5 columns: 3 interspaces, 2 colorbars

    f, axes, disps = plot_spm(zmaps, roi_dict, z_threshold=thresholds,
                              f=f, axes=axes,
                              roi_to_plot=['rIFG', 'STR', 'STN'],
                              cut_coords=[[52], [2], [-13]],
                              bg_img='/home/stevenm/Templates/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_brain.nii',
                              vmax=vmax, #colorbar=False, 
                              annotate=False, empty_plots=False, 
                              skip_all_but_last=False)
    axes[0,0].set_title(titles[0])
    axes[0,1].set_title(titles[1])
    axes[0,3].set_title(titles[2])
    axes[0,4].set_title(titles[3])
    axes[0,6].set_title(titles[4])
    axes[0,7].set_title(titles[5])

    for row in range(axes.shape[0]):
        axes[row,2].set_visible(False)
        axes[row,5].set_visible(False)
        axes[row,8].set_visible(False)
        if row in [0,1,2]:
            for col in [-3,-2,-1]:
                axes[row,col].set_visible(False)
                axes[row,col].set_visible(False)

    # for titles: https://stackoverflow.com/questions/40936729/matplotlib-title-spanning-two-or-any-number-of-subplot-columns
    ext = []
    #loop over the columns (j) and rows(i) to populate subplots
    for j in range(8):
        # save the axes bounding boxes for later use
        ext.append([axes[0,j].get_window_extent().x0, axes[0,j].get_window_extent().width ])

    # make nice
    inv = f.transFigure.inverted()
    width_left = ext[0][0]+(ext[1][0]+ext[1][1]-ext[0][0])/2.
    left_center = inv.transform( (width_left, 1) )

    width_mid = ext[3][0]+(ext[4][0]+ext[4][1]-ext[3][0])/2.
    mid_center = inv.transform( (width_mid, 1) )

    width_right = ext[6][0]+(ext[7][0]+ext[7][1]-ext[6][0])/2.
    right_center = inv.transform( (width_right, 1) )

    # set column spanning title 
    # the first two arguments to figtext are x and y coordinates in the figure system (0 to 1)
    plt.figtext(left_center[0], .93, contrast_names[0], va="center", ha="center")
    plt.figtext(mid_center[0], .93, contrast_names[1], va="center", ha="center")
    plt.figtext(right_center[0], .93, contrast_names[2], va="center", ha="center")

    # Positions in MNI-space
    axes[0,0].set_ylabel('x = 51', labelpad=50)
    axes[1,0].set_ylabel('y = 2', labelpad=50)
    axes[2,0].set_ylabel('z = -13', labelpad=50)

    # colorbar
    thrs_ = thresholds
    if isinstance(thresholds, list):
        thrs_ = thresholds[0]
    
    cbar_ax1 = f.add_subplot(gs[1,-2])
    cbar_ax1 = draw_custom_colorbar(colorbar_ax=cbar_ax1, 
                                    vmin=colorbars[0][0], vmax=colorbars[0][1],
                                    truncation_limits=colorbars[0], offset=thrs_, flip=False, **cb_kwargs)
    
    if len(colorbars) == 2:
        cbar_ax2 = f.add_subplot(gs[1,-1])
        cbar_ax2 = draw_custom_colorbar(colorbar_ax=cbar_ax2, 
                                        vmin=colorbars[1][0], vmax=colorbars[1][1],
                                        truncation_limits=(-colorbars[1][0], -colorbars[1][1]), 
                                        offset=thrs_, flip=True, **cb_kwargs)
    
        cbar_ax1.set_title(colorbar_title, rotation=90, ha='center', va='bottom', pad=16, loc='right')    
    else:
        cbar_ax1.set_title(colorbar_title, rotation=90, ha='center', va='bottom', pad=16, loc='center')    
        
    return f, axes
    # f.savefig('./glm.pdf')#, bbox_inches='tight')

def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None,
                             ax=None):
    """ 
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """
    
    if ax is None:
        ax = plt.gca()
    
    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.01
        # *** is p < 0.001
        # etc.
        if 0.01 < data < 0.05:
            text = '*'
        elif 0.001 < data < 0.01:
            text = '**'
        elif data < 0.001:
            text = '***'
#         text = ''
#         p = .05

#         while data < p:
#             text += '*'
#             p /= 10.

#             if maxasterix and len(text) == maxasterix:
#                 break

#         if len(text) == 0:
#             text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr is not None:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = ax.get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    ax.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    ax.text(*mid, text, **kwargs)
    
