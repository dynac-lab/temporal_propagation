import warnings
import config as cfg
import mne
import mne_bids
import pandas as pd
import nibabel
import numpy as np
import matplotlib.pyplot as plt
import math

session = 'iemu'
datatype = 'ieeg'
task = 'film'
acquisition = 'clinical'
fs_dir = cfg.dir_fs
bids_dir = cfg.dir_data

for n in cfg.subjects_analysis_novel + cfg.subjects_analysis_familiar:
    subject = str(n).zfill(2)
    print("Labeling " + subject)

    # Get atlas
    print("Atlas label per vertex...")
    coords_all_atlas = []
    labels_per_coords = {}
    if n == 13:
        side = 'r'
    else:
        side = 'l'
    for l_idx in cfg.atlas_BA.keys():

        # Default DK atlas
        filename = cfg.dir_fs + "sub-" + subject + '/label/' + side + 'h.PALS_B12_Brodmann.annot-' + str(l_idx).zfill(3) + '.label'

        # Try loading, but in some cases it does not exist
        try:
            L = mne.read_label(filename, subject=subject)
        except:
            print("Label " + str(l_idx) + " not found.")
            continue

        coords = L.pos
        coords_all_atlas.extend(coords)
        for vertex in coords:
            labels_per_coords[str(vertex)] = cfg.atlas_BA[l_idx]

    np.save(cfg.dir_electrode_labels + "sub-" + subject + "/" + "BA_per_vertex_sub" + subject, labels_per_coords)
    coords_all_atlas = np.asarray(coords_all_atlas)


    print("Label channels...")
    # Get final electrode names (ECoG and good)
    channels_path = mne_bids.BIDSPath(subject=subject,
                                        session=session,
                                        suffix='channels',
                                        extension='.tsv',
                                        datatype=datatype,
                                        task=task,
                                        acquisition=acquisition,
                                        root=bids_dir)
    channels = pd.read_csv(str(channels_path.match()[0]), sep='\t', header=0, index_col=None)
    names_good_electrodes = channels["name"][np.logical_and(channels['type'] == "ECOG", channels["status"] == "good")].values

    # Load electrode info
    electrodes_path = mne_bids.BIDSPath(subject=subject,
                                        session=session,
                                        suffix='electrodes',
                                        extension='.tsv',
                                        datatype=datatype,
                                        acquisition=acquisition,
                                        root=bids_dir)
    electrodes = pd.read_csv(str(electrodes_path), sep='\t', header=0, index_col=None)

    # Filter electrodes
    electrodes = electrodes[[x in names_good_electrodes for x in electrodes['name']]]

    # Get coords
    coords = electrodes[['x', 'y', 'z']].values
    x = nibabel.load(fs_dir +  'sub-' + str(subject) + '/mri/orig.mgz')
    vox_coords = np.round(mne.transforms.apply_trans(np.linalg.inv(x.affine), coords)).astype(int)
    ras_coords_electrodes = mne.transforms.apply_trans(x.header.get_vox2ras_tkr(), vox_coords)
    ras_coords_electrodes = ras_coords_electrodes / 1000

    electrode_labels = {}
    for el_coords, el_name in zip(ras_coords_electrodes,electrodes['name'].values):
        #Find which atlas vertex is closest to this electrode
        closest = coords_all_atlas[0,:]
        min_dist = math.dist(el_coords, closest)
        for at_coords in coords_all_atlas[1:,:]:
            dist = math.dist(el_coords,at_coords)
            if dist < min_dist:
                min_dist = dist
                closest = at_coords
            elif dist == min_dist:
                warnings.warn("Two vertices with the same distance found!")
        electrode_labels[el_name] = labels_per_coords[str(closest)]

    np.save(cfg.dir_electrode_labels + "sub-" + subject + "/labels_BA_sub" + subject, electrode_labels)

    # Overview of relevant BA
    print("Overview relevant BA...")
    nr_electrode_per_BA = dict((BA, 0) for BA in cfg.BA_areas)
    for el_name in electrodes['name'].values:
        BA = int(electrode_labels[el_name].split('.')[-1])
        if BA in cfg.BA_areas:
            nr_electrode_per_BA[BA] += 1
    # Print properly
    print("Subject " + str(n))
    for BA in nr_electrode_per_BA.keys():
        print("- BA" + str(BA) + ": " + str(nr_electrode_per_BA[BA]))

    # Print number of channels per ROI
    print("Low language:" + str(np.sum([nr_electrode_per_BA[BA] for BA in cfg.BA_low])))
    print("High language:" + str(np.sum([nr_electrode_per_BA[BA] for BA in cfg.BA_high])))


    fig = plt.figure(figsize=(30, 20))
    ax = fig.add_subplot(projection='3d')
    for ch_idx, channel in enumerate(electrode_labels.keys()):
        label = electrode_labels[channel]
        coords = ras_coords_electrodes[ch_idx]
        ax.scatter(coords[0], coords[1], coords[2], marker='o', s=40, c='k')
        ax.text(coords[0], coords[1], coords[2], label, 'y')
    ax.view_init(0, -180)
    plt.savefig(cfg.dir_electrode_labels + "sub-" + subject + "/labels_BA_sub" + subject)
