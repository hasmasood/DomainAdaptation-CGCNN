##############
# This additional script is to get mpids using formula and SG
##############
import os
from pymatgen.ext.matproj import MPRester
from pymatgen.io.cif import CifParser
MAPI_KEY = os.environ['MAPI_KEY']
mpr = MPRester(MAPI_KEY)

formulas = [
'CuGaSe2',
'CuGaSe2',
'CuInSe2',
#'CuIn0.5Ga0.5Se2',
'CdTe',
'ZnTe',
'ZnTe',
'CdSe',
'CsPbI3',
'CsGeI3',
# 'Sr15Ga22As32',
# 'Eu15Ga22As32',
# 'Sr15In22As32',
# 'Eu15In22As32',
# 'Sr3Ga6As8',
# 'Eu3Ga6As8',
'GaAs',
'InAs',
'InP',
'AlSb',
'GaP',
'GaSb'
]
mpids = []
for i in range (0,len(formulas)):
    mpids.append(mpr.get_materials_ids(formulas[i]))
    print(mpr.get_materials_ids(formulas[i]))
    
mpids_concat = []
for j in mpids:
    for i in range (0,len(j)):
        mpids_concat.append(j[i])
print(mpids_concat)

data = []
for i in range(0,len(mpids_concat)):
    data.append(mpr.query(mpids_concat[i], ['material_id', "pretty_formula", "spacegroup.number", "band_gap"])) 

import pandas as pd
df = pd.DataFrame(data = [])
df['mpids'] = [p[0]['material_id'] for p in data]
df['Formula'] = [p[0]['pretty_formula'] for p in data]
df['SG'] = [p[0]['spacegroup.number'] for p in data]
df['BG_MP'] = [p[0]['band_gap'] for p in data]

df.to_excel('Datasets/processed_step1/mpids_from_formulas_sg.xlsx')
