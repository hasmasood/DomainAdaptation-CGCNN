# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 12:35:53 2021

@author: z5022637
"""
import pandas as pd
from pymatgen.ext.matproj import MPRester
from pymatgen.io.cif import CifParser

MAPI_KEY = 'saVWIDcr59SCl2L8'
mpr = MPRester(MAPI_KEY)

file = 'processed_m2_Wan.xlsx'

#Get structures from xl    
df = pd.ExcelFile(file).parse('Sheet1')
QUERY = df['mpids'].values.tolist()
        
print('Total entries:', len(QUERY))

QUERY_sorted = []
for i in range(0,len(QUERY)):
    if QUERY[i] not in QUERY_sorted:
        QUERY_sorted.append(QUERY[i])
QUERY = QUERY_sorted

print('Total sorted entries after removing duplicates:', len(QUERY))

structures_cif = []
for i in range(0,len(QUERY)):
    structures_cif.append(mpr.get_data(QUERY[i], data_type = '', prop = 'cif').get('cif'))
print("No. of structures:",len(structures_cif))


#Save structures in cif file
for i in range(0,len(structures_cif)):
    with open('cif/{}.cif'.format(QUERY[i]),'w') as output:
        output.write(structures_cif[i])





################################################################################
################################################################################
################################################################################
################################################################################
prop_bg = []
for i in range(0,len(QUERY)):
    prop_bg.append(mpr.get_data(QUERY[i], prop = 'band_gap'))
    #print('MP-ID =',prop_bg[i][0]['material_id'], 'Band Gap =',prop_bg[i][0]['band_gap'])
    
df_prop_bg = pd.DataFrame(data =[])
df_prop_bg['MP-ID'] = [p[0]['material_id'] for p in prop_bg]
df_prop_bg['BG'] = [p[0]['band_gap'] for p in prop_bg]

df_prop_bg.to_csv("prop.csv", index=False)


#
## For Filtering structures according to BG


import pandas as pd
from pymatgen.ext.matproj import MPRester
from pymatgen.io.cif import CifParser

MAPI_KEY = 'saVWIDcr59SCl2L8'
mpr = MPRester(MAPI_KEY)


#Get structures from xl    
df = pd.ExcelFile('SC.xlsx').parse('Sheet1')
QUERY = df['MaterialsId'].values.tolist()
        
print('Total entries:', len(QUERY))

bg_threshold = 0.5 

short = []
structures_cif = []
prop_bg = []
for i in range(0,len(QUERY)):
    if mpr.get_data(QUERY[i], prop = 'band_gap')[0]['band_gap'] > bg_threshold:
            short.append(QUERY[i])
            structures_cif.append(mpr.get_data(QUERY[i], data_type = '', prop = 'cif').get('cif'))
            prop_bg.append(mpr.get_data(QUERY[i], prop = 'band_gap'))

QUERY = short      
print("No. of structures:",len(structures_cif))

#Save structures in cif file
for i in range(0,len(structures_cif)):
    with open('Filtered/cif/{}.cif'.format(QUERY[i]),'w') as output:
        output.write(structures_cif[i])
#Write BG
df_prop_bg = pd.DataFrame(data =[])
df_prop_bg['MP-ID'] = [p[0]['material_id'] for p in prop_bg]
df_prop_bg['BG'] = [p[0]['band_gap'] for p in prop_bg]
df_prop_bg.to_csv("Filtered/prop.csv", index=False)

#
################################################################################       