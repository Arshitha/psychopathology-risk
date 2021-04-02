---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.10.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Project Week: Specificity of Structural Brain Markers of Psychopathology Risk
### Data exploration of the ABCD dataset. The final output of this notebook is a clean dataset file for the Linear Mixed Models in R. 
1. Import libraries
1. Gather all the data elements from the ABCD text files
1. Specify variables of interest
1. Create a dataframe with variables of interest
1. Remove duplicates and apply exclusion criteria
1. Create addiction variables by combining alcohol and drug abuse pathology
1. Count number of children in each parental history psychopathology group 
1. Save clean dataframe for LMM

### 1. Import libraries

just checking
```python
import pandas as pd # to read/manipulate/write data from files
import numpy as np # to manipulate data/generate random numbers
import plotly.express as px # interactive visualizations
import seaborn as sns # static visualizations
import matplotlib.pyplot as plt # fine tune control over visualizations

from pathlib import Path # represent and interact with directories/folders in the operating system
from collections import namedtuple # structure data in an easy to consume way

import requests # retrieve data from an online source
```

### 2. Gather all the data elements from all the txt files

The ABCD3 folder contains a number of tab-delimited text files with ABCD data.
We will first collect all of these files and then load them into pandas DataFrames
to compile and access the data.
```python
# save directory we downloaded the ABCD data to `data_path`
data_path = Path("/shared/project-psychopathology-risk/inputs/data/dataset1/")
# glob (match) all text files in the `data_path` directory
files = sorted(data_path.glob("*.txt"))
```

After collecting the files, we need to extract information about
the data structures and the data elements.
The data structures are the text file names (e.g., `abcd_abcls01`)
which indicate the type of data stored inside the text file.
The data elements are the column names inside the tab delimited
text file.

The data structure and data element names are condensed to make working
with them programmatically easier, but it is difficult for a human
to interpret what `abcd_abcls01` means.
So in addition to aggregating data structure and data element names
together, we are also collecting their metadata to have a human readable description
of their condensed names.

The data element metadata is located in the data structure files themselves
as the second row of the file, however, the data structure metadata was not downloaded
necessitating a query to the NDA website to retrieve the human readable
description of the data structure (using `requests`).

Finally, since we are only interested in the baseline measures, we need to keep track of
what names are given to each event in each of the data structures.
```python
# We store the info in 4 different Python datatypes
data_elements = []
data_structures = {}
event_names = set()
StructureInfo = namedtuple("StructureInfo", field_names=["description", "eventnames"])

for text_file in files:
    # Extract data structure from filename
    data_structure = Path(text_file).name.split('.txt')[0]
    
    # Read the data structure and capture all the elements from the file
    # Note this could have been done using the data returned from the NDA API
    # We are using pandas to read both the first and second rows of the file as the header
    # Note: by convention dataframe variables contain `df` in the name.
    data_structure_df = pd.read_table(text_file, header=[0, 1], nrows=0)
    for data_element, metadata in data_structure_df.columns.values.tolist():
        data_elements.append([data_element, metadata, data_structure])

    
    # (Optional) Retrieve the eventnames in each structure. Some structures were only collected
    # at baseline while others were collected at specific or multiple timepoints
    events_in_structure = None
    if any(['eventname' == data_element for data_element in data_structure_df.columns.levels[0]]):
        # Here we are skipping the 2nd row of the file containing description using skiprows
        possible_event_names_df = pd.read_table(text_file, skiprows=[1], usecols=['eventname'])
        events_in_structure = possible_event_names_df.eventname.unique().tolist()
        event_names.update(events_in_structure)

    # (Optional) Retrieve the title for the structure using the NDA API
    rinfo = requests.get(f"https://nda.nih.gov/api/datadictionary/datastructure/{data_structure}").json()
    data_structures[data_structure] = StructureInfo(description=rinfo["title"] if "title" in rinfo else None,
                                                    eventnames=events_in_structure)

# Convert to a Pandas dataframe
data_elements_df = pd.DataFrame(data_elements, columns=["element", "description", "structure"])
```

```python
## Uncomment next line to save data elements to a tab-separated file
# data_elements_df.to_csv("/shared/project-psychopathology-risk/outputs/exploration/data_elements.tsv", sep="\t", index=None)
```
### 3. Specify variables of interest

```python
# Variables for dataframe
common = ['subjectkey', 'interview_age', 'interview_date', 'eventname', 'sex']
nested = ['rel_family_id', 'mri_info_deviceserialnumber', 'site_id_l','acs_raked_propensity_score']
puberty = ['pds_p_ss_female_category', 'pds_p_ss_male_category']

#freesurfer subcortical volume in mm^3 of ASEG ROI
scvol = ['smri_vol_scs_aal', 'smri_vol_scs_aar', 'smri_vol_scs_amygdalalh', 'smri_vol_scs_amygdalarh', 
         'smri_vol_scs_caudatelh', 'smri_vol_scs_caudaterh', 'smri_vol_scs_hpuslh', 'smri_vol_scs_hpusrh', 
         'smri_vol_scs_pallidumlh', 'smri_vol_scs_pallidumrh', 'smri_vol_scs_putamenlh', 'smri_vol_scs_putamenrh',
         'smri_vol_scs_tplh', 'smri_vol_scs_tprh', 'smri_vol_scs_intracranialv', 'smri_vol_scs_subcorticalgv']

# parental psychopathology 
famhx_moth = ['famhx_ss_moth_prob_dprs_p','famhx_ss_moth_prob_alc_p', 'famhx_ss_moth_prob_dg_p', 
              'famhx_ss_moth_prob_ma_p', 'famhx_ss_moth_prob_nrv_p']
famhx_fath = ['famhx_ss_fath_prob_dprs_p','famhx_ss_fath_prob_alc_p', 'famhx_ss_fath_prob_dg_p', 
              'famhx_ss_fath_prob_ma_p', 'famhx_ss_fath_prob_nrv_p']
famhx_momdad = ['famhx_ss_momdad_dprs_p','famhx_ss_momdad_alc_p', 'famhx_ss_momdad_dg_p', 
                'famhx_ss_momdad_ma_p', 'famhx_ss_momdad_nrv_p']
famhx_parent =['famhx_ss_parent_dprs_p','famhx_ss_parent_alc_p', 'famhx_ss_parent_dg_p', 
               'famhx_ss_parent_ma_p', 'famhx_ss_parent_nrv_p']
# quality control (exclusion criteria)
qc = ["iqc_t1_ok_ser", "fsqc_qc", "mrif_score", 'demo_prim', 'famhx_ss_momdad_vs_p']

piagliaccio = ["race_ethnicity", "demo_comb_income_v2", "demo_prnt_ed_v2"]
#pagliaccio2 = ["demo_prnt_marital_v2", "anthro_1_height_in", "ksads_1_842_p", "ksads_1_1_t", "cbcl_q01_p", "cbcl_scr_syn_anxdep_r"]

#freesurfer cortical volume
cortvol = ["smri_vol_cdk_banksstslh", "smri_vol_cdk_banksstsrh", "smri_vol_cdk_cdacatelh", "smri_vol_cdk_cdacaterh", 
           "smri_vol_cdk_cdmdfrlh", "smri_vol_cdk_cdmdfrrh", "smri_vol_cdk_cuneuslh", "smri_vol_cdk_cuneusrh", 
           "smri_vol_cdk_ehinallh", "smri_vol_cdk_ehinalrh", 'smri_vol_cdk_frpolelh', 'smri_vol_cdk_frpolerh', 
           'smri_vol_cdk_fusiformlh', 'smri_vol_cdk_fusiformrh', 'smri_vol_cdk_ifpllh', 'smri_vol_cdk_ifplrh', 
           'smri_vol_cdk_iftmlh', 'smri_vol_cdk_iftmrh', 'smri_vol_cdk_ihcatelh', 'smri_vol_cdk_ihcaterh',
           'smri_vol_cdk_insulalh', 'smri_vol_cdk_insularh','smri_vol_cdk_linguallh', 'smri_vol_cdk_lingualrh', 
           'smri_vol_cdk_lobfrlh', 'smri_vol_cdk_lobfrrh', 'smri_vol_cdk_locclh', 'smri_vol_cdk_loccrh',
           'smri_vol_cdk_mdtmlh', 'smri_vol_cdk_mdtmrh', 'smri_vol_cdk_mobfrlh', 'smri_vol_cdk_mobfrrh',
           'smri_vol_cdk_paracnlh', 'smri_vol_cdk_paracnrh', 'smri_vol_cdk_parahpallh', 'smri_vol_cdk_parahpalrh', 
           'smri_vol_cdk_parsobislh', 'smri_vol_cdk_parsobisrh', 'smri_vol_cdk_parsopclh', 'smri_vol_cdk_parsopcrh', 
           'smri_vol_cdk_parstgrislh', 'smri_vol_cdk_parstgrisrh', 'smri_vol_cdk_pclh', 'smri_vol_cdk_pcrh', 
           'smri_vol_cdk_pericclh', 'smri_vol_cdk_periccrh', 'smri_vol_cdk_postcnlh', 'smri_vol_cdk_postcnrh', 
           'smri_vol_cdk_precnlh', 'smri_vol_cdk_precnrh', 'smri_vol_cdk_ptcatelh', 'smri_vol_cdk_ptcaterh', 
           'smri_vol_cdk_rracatelh', 'smri_vol_cdk_rracaterh', 'smri_vol_cdk_rrmdfrlh', 'smri_vol_cdk_rrmdfrrh', 
           'smri_vol_cdk_smlh', 'smri_vol_cdk_smrh', 'smri_vol_cdk_sufrlh', 'smri_vol_cdk_sufrrh', 
           'smri_vol_cdk_supllh', 'smri_vol_cdk_suplrh', 'smri_vol_cdk_sutmlh', 'smri_vol_cdk_sutmrh', 
           'smri_vol_cdk_tmpolelh', 'smri_vol_cdk_tmpolerh', 'smri_vol_cdk_trvtmlh', 'smri_vol_cdk_trvtmrh']

cortthick = ["smri_thick_cdk_banksstslh", "smri_thick_cdk_banksstsrh", "smri_thick_cdk_cdacatelh", "smri_thick_cdk_cdacaterh", 
           "smri_thick_cdk_cdmdfrlh", "smri_thick_cdk_cdmdfrrh", "smri_thick_cdk_cuneuslh", "smri_thick_cdk_cuneusrh", 
           "smri_thick_cdk_ehinallh", "smri_thick_cdk_ehinalrh", 'smri_thick_cdk_frpolelh', 'smri_thick_cdk_frpolerh', 
           'smri_thick_cdk_fusiformlh', 'smri_thick_cdk_fusiformrh', 'smri_thick_cdk_ifpllh', 'smri_thick_cdk_ifplrh', 
           'smri_thick_cdk_iftmlh', 'smri_thick_cdk_iftmrh', 'smri_thick_cdk_ihcatelh', 'smri_thick_cdk_ihcaterh',
           'smri_thick_cdk_insulalh', 'smri_thick_cdk_insularh','smri_thick_cdk_linguallh', 'smri_thick_cdk_lingualrh', 
           'smri_thick_cdk_lobfrlh', 'smri_thick_cdk_lobfrrh', 'smri_thick_cdk_locclh', 'smri_thick_cdk_loccrh',
           'smri_thick_cdk_mdtmlh', 'smri_thick_cdk_mdtmrh', 'smri_thick_cdk_mobfrlh', 'smri_thick_cdk_mobfrrh',
           'smri_thick_cdk_paracnlh', 'smri_thick_cdk_paracnrh', 'smri_thick_cdk_parahpallh', 'smri_thick_cdk_parahpalrh', 
           'smri_thick_cdk_parsobislh', 'smri_thick_cdk_parsobisrh', 'smri_thick_cdk_parsopclh', 'smri_thick_cdk_parsopcrh', 
           'smri_thick_cdk_parstgrislh', 'smri_thick_cdk_parstgrisrh', 'smri_thick_cdk_pclh', 'smri_thick_cdk_pcrh', 
           'smri_thick_cdk_pericclh', 'smri_thick_cdk_periccrh', 'smri_thick_cdk_postcnlh', 'smri_thick_cdk_postcnrh', 
           'smri_thick_cdk_precnlh', 'smri_thick_cdk_precnrh', 'smri_thick_cdk_ptcatelh', 'smri_thick_cdk_ptcaterh', 
           'smri_thick_cdk_rracatelh', 'smri_thick_cdk_rracaterh', 'smri_thick_cdk_rrmdfrlh', 'smri_thick_cdk_rrmdfrrh', 
           'smri_thick_cdk_smlh', 'smri_thick_cdk_smrh', 'smri_thick_cdk_sufrlh', 'smri_thick_cdk_sufrrh', 
           'smri_thick_cdk_supllh', 'smri_thick_cdk_suplrh', 'smri_thick_cdk_sutmlh', 'smri_thick_cdk_sutmrh', 
           'smri_thick_cdk_tmpolelh', 'smri_thick_cdk_tmpolerh', 'smri_thick_cdk_trvtmlh', 'smri_thick_cdk_trvtmrh', 
           'smri_thick_cdk_meanlh', 'smri_thick_cdk_meanrh', 'smri_thick_cdk_mean']


data_elements_of_interest = nested + puberty + scvol + famhx_moth + famhx_fath + famhx_momdad + famhx_parent + qc + piagliaccio + cortvol + cortthick

```

#### Find the data structures that contain the data elements

The `data_elements_of_interest` above tell us what data elements we wish to analyze,
but they do not provide information about which data structures the data elements are located.
But do not fret, we created `data_elements_df` to match data elements with their respective data structures,
giving us the ability to find the data structure associated with each data element of interest.

```python
structures2read = {}
for element in data_elements_of_interest:
    item = data_elements_df.query(f"element == '{element}'").structure.values[0]
    if item not in structures2read:
        structures2read[item] = []
    structures2read[item].append(element)
structures2read
```

### 4. Create a dataframe with the variables of interest
Now we have the data structures that contain the data elements of interest in `structures2read`,
a dictionary whose keys are the data structures and whose values are the data elements of interest
within that data structure.
Here we load the data structures into python with the variables (elements) of interest. 
```python
all_df = None
for structure, elements in structures2read.items():
    data_structure_filtered_df = pd.read_table(data_path / f"{structure}.txt", skiprows=[1], low_memory=False, usecols=common + elements)
    data_structure_filtered_df = data_structure_filtered_df.query("eventname == 'baseline_year_1_arm_1'")
    if all_df is None:
        all_df =  data_structure_filtered_df[["subjectkey", "interview_date", "interview_age", "sex", "eventname"] + elements]
    else:
        all_df = all_df.merge( data_structure_filtered_df[['subjectkey'] + elements], how='outer')
```

```python
all_df.shape, all_df.subjectkey.unique().shape
```

### 5. Remove duplicates and apply exclusion criteria
```python
# Remove duplicates
all_df = all_df.drop_duplicates(subset=['subjectkey'])
all_df.shape, all_df.subjectkey.unique().shape
```

```python
#Copy dataframe/backup
df1 = all_df.copy()
df1.shape #output is number of children x number of variables
```
##### Exclusion criteria 
1. iqc_t1_ok_ser: quality T1 scans == 0 
1. fsqc_qc: quality control freesurfer outputs. 0 = reject; 1 = accept
1. mrif_score: incidental findings from neuroradiological read of the sMRI. 0 = reject; 1 = accept
1. demo_prim: if parental report was not based on biological parent. Exclude value >2
1. If either parent endorsed visions of others spying/plotting problems. Exclude value == 1
```python
# Count subjects that will be excluded per variable for method section
(df1['iqc_t1_ok_ser'] == 0).astype(int).sum(axis=0)        #40 subjects
(df1['fsqc_qc'] == 0).astype(int).sum(axis=0)              #475 subjects
(df1['mrif_score'] ==0).astype(int).sum(axis=0)            #47
(df1['demo_prim'] > 2).astype(int).sum(axis=0)             #560 subjects
(df1['famhx_ss_momdad_vs_p'] == 1).astype(int).sum(axis=0) #241 subjects
```
```python
#Remove rows that fullfill at least 1 of the exclusion criteria
indexNames = df1[(df1['iqc_t1_ok_ser'] == 0) | (df1['fsqc_qc'] == 0) | (df1['mrif_score'] == 0) | (df1['demo_prim'] > 2) | (df1['famhx_ss_momdad_vs_p'] == 1)].index   
df1.drop(indexNames , inplace=True)    # Delete these row indexes from dataFrame
df1.shape #output = number of children x number of variables

#copy/backup
df2 = df1.copy()
df2.shape
```
### 6. Create addiction variables based on alcohol and drug abuse
```python
df2['famhx_ss_moth_addiction'] = np.where((df2.famhx_ss_moth_prob_alc_p == 1) | (df2.famhx_ss_moth_prob_dg_p == 1), 1.0, 0.0)
df2['famhx_ss_fath_addiction'] = np.where((df2.famhx_ss_fath_prob_alc_p == 1) | (df2.famhx_ss_fath_prob_dg_p == 1), 1.0, 0.0)
df2['famhx_ss_momdad_addiction'] = np.where((df2.famhx_ss_momdad_alc_p == 1) | (df2.famhx_ss_momdad_dg_p == 1), 1.0, 0.0)

# Not sure how to code the famhx_ss_parent_addiction with 6 different values (in particularly the negative values) 
# For now I created a binary variable that shows whether both parents have a problem or not.
df2['famhx_ss_parent_addiction_bin'] = np.where((df2.famhx_ss_parent_alc_p == 3) | (df2.famhx_ss_parent_dg_p == 3), 1.0, 0.0) #both parent have addiction problem (0 = no, 1 = yes)
df2['famhx_ss_parent_dprs_bin'] = np.where((df2.famhx_ss_parent_dprs_p == 3), 1.0, 0.0)
df2['famhx_ss_parent_ma_bin'] = np.where((df2.famhx_ss_parent_ma_p == 3), 1.0, 0.0)
df2['famhx_ss_parent_nrv_bin'] = np.where((df2.famhx_ss_parent_nrv_p == 3), 1.0, 0.0)

#explore addiction counts
s1 = df2.groupby(['famhx_ss_moth_addiction', 'famhx_ss_moth_prob_alc_p', 'famhx_ss_moth_prob_dg_p']).size().reset_index().rename(columns={0:'count'})  #addiction mother
s2 = df2.groupby(['famhx_ss_fath_addiction', 'famhx_ss_fath_prob_alc_p', 'famhx_ss_fath_prob_dg_p']).size().reset_index().rename(columns={0:'count'})  #addiction father
s3 = df2.groupby(['famhx_ss_momdad_addiction', 'famhx_ss_momdad_alc_p', 'famhx_ss_momdad_dg_p']).size().reset_index().rename(columns={0:'count'})  #addiction either mother or father
s4 = df2.groupby(['famhx_ss_parent_addiction_bin', 'famhx_ss_parent_alc_p', 'famhx_ss_parent_dg_p']).size().reset_index().rename(columns={0:'count'})  #addiction both parents
```

```python
s1 #view count table - insert s1-s4
```

```python
#save data - note that this file contains many variables that are not needed for the LMM. 
#copy/backup and remove variable columns we do not need anymore including 'famhx_ss_*_alc_p', 'famhx_ss_*_dg_p'])
df3 = df2.copy()
df3.to_csv("/shared/project-psychopathology-risk/outputs/exploration/PsychRisk_data.tsv", sep="\t", index=None)# 
```

### 7. Exploration number of children in each parental history psychopathology group

```python
# Remove columns that are no longer needed including alcohol and drug abuse history and quality control variables
df3 = df3.drop(columns=['famhx_ss_moth_prob_alc_p', 'famhx_ss_fath_prob_alc_p', 'famhx_ss_momdad_alc_p', 'famhx_ss_parent_alc_p', 
                  'famhx_ss_moth_prob_dg_p', 'famhx_ss_fath_prob_dg_p', 'famhx_ss_momdad_dg_p', 'famhx_ss_parent_dg_p',
                 'famhx_ss_momdad_vs_p', 'iqc_t1_ok_ser', 'fsqc_qc', 'demo_prim', 'mrif_score'])
```

```python
#Count data
filter_col = [col for col in df3 if col.startswith('famhx_')]
fh_df = df3[filter_col] #create dataframe with parental history psychopathology variables          
fh_df.apply(pd.Series.value_counts)        #counts
```

```python
#Count data and comorbidities
t1 = fh_df.groupby(['famhx_ss_moth_prob_dprs_p', 'famhx_ss_moth_addiction','famhx_ss_moth_prob_ma_p', 'famhx_ss_moth_prob_nrv_p']).size().reset_index().rename(columns={0:'count'})  #pathology mother
t2 = fh_df.groupby(['famhx_ss_fath_prob_dprs_p', 'famhx_ss_fath_addiction','famhx_ss_fath_prob_ma_p', 'famhx_ss_fath_prob_nrv_p']).size().reset_index().rename(columns={0:'count'})  #pathology mother
t3 = fh_df.groupby(['famhx_ss_momdad_dprs_p', 'famhx_ss_momdad_addiction','famhx_ss_momdad_ma_p', 'famhx_ss_momdad_nrv_p']).size().reset_index().rename(columns={0:'count'})  #pathology mother
t4 = fh_df.groupby(['famhx_ss_parent_dprs_p', 'famhx_ss_parent_addiction_bin','famhx_ss_parent_ma_p', 'famhx_ss_parent_nrv_p']).size().reset_index().rename(columns={0:'count'})  #pathology mother

#uncomment to save file to .csv
#t1.to_csv('/shared/project-psychopathology-risk/outputs/exploration/count_mother.csv')
#t2.to_csv('/shared/project-psychopathology-risk/outputs/exploration/count_father.csv')
#t3.to_csv('/shared/project-psychopathology-risk/outputs/exploration/count_momdad.csv')
#t4.to_csv('/shared/project-psychopathology-risk/outputs/exploration/count_parents.csv')
```

```python
t3 #print table t3
```

```python
df3.describe(include="all")
```

### 8. Save clean dataframe for Linear Mixed Models

```python
df3.to_csv("/shared/project-psychopathology-risk/outputs/exploration/PsychRisk3.tsv", sep="\t", index=None) #input LMM
```

```python

```
