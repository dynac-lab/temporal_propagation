import numpy as np
import sys


### Directories
root = "XXX"
dir_data = root + "data/"
dir_fig = root + "figures/"
dir_analysis_results = root + "figures/Analysis/"
dir_GSBS = root + "GSBS/"
dir_preGSBS = root + "preGSBS/"
dir_fs = dir_data + "freesurfer/"
dir_electrode_labels = root + "electrode_labels/"
dir_annotations = root + "data/stimuli/annotations/"


### Parameters
fs = 512
GSBS_fs = 64
nr_permutations = 1000
nr_permutations_coinflip = 10000
speech_blocks = np.arange(1,13,2)
music_blocks = np.arange(0,13,2)
band = 'preproc'
gaussian_sd = GSBS_fs * 0.1 # gaussian_sd in number of timepoints. Final factor (in seconds) determined by inspection various options with subject 5 clauses, 1000 permutations areound the peak


### ROI definitions
BA_areas = [20, 21, 22, 38, 39, 40, 41, 42, 44, 45, 46,47]
BA_low = [20,21,22,41,42]
BA_high = [38,39,40,44,45,46,47]
atlas_BA =  { 2:  "Brodmann.8",
  3:  "Brodmann.6",
  4:  "Brodmann.4",
  5:  "Brodmann.9",
  7:  "Brodmann.3",
  8:  "Brodmann.1",
  9:  "Brodmann.5",
 10:  "Brodmann.7",
 11:  "Brodmann.2",
 12:  "Brodmann.31",
 14:  "Brodmann.40",
 15:  "Brodmann.44",
 16:  "Brodmann.45",
 18:  "Brodmann.23",
 19:  "Brodmann.39",
 20:  "Brodmann.43",
 24:  "Brodmann.19",
 26:  "Brodmann.47",
 27:  "Brodmann.41",
 28:  "Brodmann.30",
 29:  "Brodmann.22",
 30:  "Brodmann.42",
 32:  "Brodmann.21",
 33:  "Brodmann.38",
 35:  "Brodmann.37",
 38:  "Brodmann.20",
 41:  "Brodmann.32",
 42:  "Brodmann.24",
 43:  "Brodmann.10",
 44:  "Brodmann.25",
 45:  "Brodmann.11",
 46:  "Brodmann.46",
 47:  "Brodmann.17",
 48:  "Brodmann.18",
 49:  "Brodmann.27",
 50:  "Brodmann.36",
 51:  "Brodmann.35",
 52:  "Brodmann.28",
 53:  "Brodmann.29",
 54:  "Brodmann.26"
}

atlas_BN = {0: 'Unknown',
1: 'A8m_L',
3: 'A8dl_L',
5: 'A9l_L',
7: 'A6dl_L',
9: 'A6m_L',
11:'A9m_L',
13: 'A10m_L',
15: 'A9/46d_L',
17: 'IFJ_L',
19: 'A46_L',
21: 'A9/46v_L',
23: 'A8vl_L',
25: 'A6vl_L',
27: 'A10l_L',
29: 'A44d_L',
31: 'IFS_L',
33: 'A45c_L',
35: 'A45r_L',
37: 'A44op_L',
39: 'A44v_L',
41: 'A14m_L',
43: 'A12/47o_L',
45: 'A11l_L',
47: 'A11m_L',
49: 'A13_L',
51: 'A12/47l_L',
53: 'A4hf_L',
55: 'A6cdl_L',
57: 'A4ul_L',
59: 'A4t_L',
61: 'A4tl_L',
63: 'A6cvl_L',
65: 'A1/2/3ll_L',
67: 'A4ll_L',
69: 'A38m_L',
71: 'A41/42_L',
73: 'TE1.0/TE1.2_L',
75: 'A22c_L',
77: 'A38l_L',
79: 'A22r_L',
81: 'A21c_L',
83: 'A21r_L',
85: 'A37dl_L',
87: 'aSTS_L',
89: 'A20iv_L',
91: 'A37elv_L',
93: 'A20r_L',
95: 'A20il_L',
97: 'A37vl_L',
99: 'A20cl_L',
101:'A20cv_L',
103:'A20rv_L',
105:'A37mv_L',
107:'A37lv_L',
109:'A35/36r_L',
111:'A35/36c_L',
113: 'TL_L',
115: 'A28/34_L',
117: 'TI_L',
119: 'TH_L',
121: 'rpSTS_L',
123: 'cpSTS_L',
125: 'A7r_L',
127: 'A7c_L',
129: 'A5l_L',
131: 'A7pc_L',
133: 'A7ip_L',
135: 'A39c_L',
137: 'A39rd_L',
139: 'A40rd_L',
141: 'A40c_L',
143: 'A39rv_L',
145: 'A40rv_L',
147: 'A7m_L',
149: 'A5m_L',
151: 'dmPOS_L',
153: 'A31_L',
155: 'A1/2/3ulhf_L',
157: 'A1/2/3tonIa_L',
159: 'A2_L',
161: 'A1/2/3tru_L',
163: 'G_L',
165: 'vIa_L',
167: 'dIa_L',
169: 'vId/vIg_L',
171: 'dIg_L',
173: 'dId_L',
175: 'A23d_L',
177: 'A24rv_L',
179: 'A32p_L',
181: 'A23v_L',
183: 'A24cd_L',
185: 'A23c_L',
187: 'A32sg_L',
189: 'cLinG_L',
191: 'rCunG_L',
193: 'cCunG_L',
195: 'rLinG_L',
197: 'vmPOS_L',
199: 'mOccG_L',
201: 'V5/MT+_L',
203: 'OPC_L',
205: 'iOccG_L',
207: 'msOccG_L',
209: 'lsOccG_L'
}

### Subjects ###
# Good subjects divided into two groups: those who did fMRI as well and those who did not
good_subjects_ECoG = [3,5,12,26,36,48,54,57,59]
good_subjects_fMRI = [7,22,45,46,51,55,60]
bad_language_subjects = [14,17,28,38,43] # 30 (remove 30 because none of the channels were in the ROIs)
all_subjects = good_subjects_ECoG + good_subjects_fMRI + bad_language_subjects
subjects_rh = [17,28,43]
subjects_with_HD = [45] #36  # According to the pictures, also 60, but there is no sepearte data.
subjects_analysis_novel = [5,12, 26, 36, 54, 59] # Removed some subjects based on visual inspection
subjects_analysis_familiar = [22, 45, 46, 51, 55]

# Bad channels based on pre-GSBS visual inspection
bad_channels = {
    '03': [], # dropped subject; visualization purpose only
    '48': [], # dropped subject; visualization purpose only
    '57': [], # dropped subject; visualization purpose only
    '60': [], # dropped subject; visualization purpose only
    '05': [],
    '12': [],
    '26': ['T06', 'T28', 'sOc6', 'sTa7', 'sTa3', 'AT35', 'AT33', 'T15', 'sTv4', 'sTa6', 'P58', 'T27'],
    '59': ['OT15'],
    '22': [],
    '45': ['F46', 'F43','vT8'],
    'HDgrid_45': [],
    '46': [],
    '51': [],
    '55': ['sTv7'],
    '36': [],
    '07': ['C01','C02','C09','C10','C11','C12',],
    '54': [],
    '13': ['P46', 'OB5', 'P07', 'P08', 'P16'],
    '16': ['FM01','FM02','FM03','FM04','FM05','FM06','FM07','FM08','FM09','FM10','FM11','FM12','FM13','FM14','FM15','FM16', 'FL01','FL02','FL03','FL04','FL05','FL06','FL07','FL08','FL09','FL10','FL11','FL12','FL13','FL14','FL15','FL16',],
    '18': [],
    '25': ['HF01'],
    '27': [],
    '58': [],
    '61': [],
}

preproc_per_subject = {
    '03': 'infomax', # excluded subject; visualization purpose only
    '48': 'infomax', # excluded subject; visualization purpose only
    '57': 'infomax', # excluded subject; visualization purpose only
    '60': 'infomax', # excluded subject; visualization purpose only
    '05': 'car',
    '12': 'infomax',
    '26': 'infomax',
    '59': 'infomax',
    '22': 'infomax',
    '45': 'infomax',
    'HDgrid_45': 'infomax',
    '46': 'car',
    '51': 'infomax',
    '54': 'infomax',
    '55': 'infomax',
    '36': 'car',
    '07': 'infomax',
    '16':'infomax',
    '18':'infomax',
    '25':'car',
    '27':'infomax',
    '58':'infomax',
    '61':'infomax',
}


# Load adjusted number of strengths
adjusted_nr_of_states = {}
for ROI in ['low', 'high']:
    ROI_name = 'ROI-' + ROI
    try:
        adjusted_nr_of_states[ROI] = np.load(dir_analysis_results + 'GSBS/GSBS adjustments/' + 'states_per_subject_' + ROI_name + '.npy', allow_pickle=True).item()
    except:
        print("Looks like adjusted number of states has not been computed yet")
        adjusted_nr_of_states[ROI] = None

# Parameters speech analysis
speech_annotation_categories = ['words','clauses']
speech_delays = np.arange(0,0.601,1/GSBS_fs)
ROI_delays = np.arange(- 0.609375, 0.61, 1/GSBS_fs) # Not exactly starting at -0.6 to have 0.0 in there


### Plot settings
fig_width = {
    'one_column': 8/2.54,
    'two_column': 17.8/2.54
}

colorblind_scheme={
    'black': '#000000',
    'orange': '#E69F00',
    'lightblue': '#56B4E9',
    'green': '#009E73',
    'yellow': '#F0E442',
    'blue': '#0072B2',
    'red': '#D55E00',
    'purple': '#CC79A7',
    'brown': '#753E00' #Costum
}
colors_ROI = {
    'low': colorblind_scheme['orange'],
    'high': colorblind_scheme['blue']
}
grid_color = 'darkgray'