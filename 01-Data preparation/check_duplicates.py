# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 15:31:05 2022

@author: z5022637
"""
import pandas as pd

file_1 = '.xlsx'
file_2 = '.xlsx'
file_3 = '.xlsx'
file_4 = '.xlsx'

#save_file_m2inm1 = 'm2inm1.csv'

# import csv
# with open(file_1, newline='') as f:
#     reader = csv.reader(f)
#     m1 = []
#     for row in reader:
#         m1.append(row[0])
# with open(file_2, newline='') as f:
#     reader = csv.reader(f)
#     m2 = []
#     for row in reader:
#         m2.append(row[0])  


m1 = pd.ExcelFile(file_1).parse('Sheet1')['mpids'].values.tolist()
m2 = pd.ExcelFile(file_2).parse('Sheet1')['mpids'].values.tolist()
validation = pd.ExcelFile(file_3).parse('Sheet1')['mpids'].values.tolist()
enrich_a = pd.ExcelFile(file_4).parse('Sheet1')['mpids'].values.tolist()

norep_m2 = list(dict.fromkeys(m2))

#M2 in M1      
m2inm1 = []
for i in m2:
    if i in m1:
        m2inm1.append(i)

norep_m2inm1 = list(dict.fromkeys(m2inm1))
norep_m2inm1 = pd.DataFrame(norep_m2inm1)

#Validation in M2        
valinm2 = []
for i in validation:
    if i in m2:
        valinm2.append(i)

norep_valinm2 = list(dict.fromkeys(valinm2))
norep_valinm2 = pd.DataFrame(norep_valinm2) 

#Validation in enrich_a   
valinenrich_a = []
for i in validation:
    if i in enrich_a:
        valinenrich_a.append(i)

norep_valinenrich_a = list(dict.fromkeys(valinenrich_a))
norep_valinenrich_a = pd.DataFrame(norep_valinenrich_a)

#enrich_a in M2        
enrich_a_inm2 = []
for i in enrich_a:
    if i in m2:
        enrich_a_inm2.append(i)

norep_enrich_a_inm2 = list(dict.fromkeys(enrich_a_inm2))
norep_enrich_a_inm2 = pd.DataFrame(norep_enrich_a_inm2)  

#enrich_a in M1        
enrich_a_inm1 = []
for i in enrich_a:
    if i in m1:
        enrich_a_inm1.append(i)

norep_enrich_a_inm1 = list(dict.fromkeys(enrich_a_inm1))
norep_enrich_a_inm1 = pd.DataFrame(norep_enrich_a_inm1)  



#Validation in M1       
m1_processed = pd.ExcelFile('processed_M1.xlsx').parse('Sheet1')['mpids'].values.tolist()
validation_processed = pd.ExcelFile('processed_validation.xlsx').parse('Sheet1')['mpids'].values.tolist()

valinm1_processed = []
for i in validation_processed:
    if i in m1_processed:
        valinm1_processed.append(i)

#Save files
# norep_m2inm1.to_csv(save_file_m2inm1, index=False, header=False)


