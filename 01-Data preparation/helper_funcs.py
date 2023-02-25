# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 12:02:08 2022

@author: z5022637
"""
import pandas as pd
import re

def check_icsd(df):
    import pymatgen
    import os
    MAPI_KEY = os.environ['MAPI_KEY']
    from pymatgen.ext.matproj import MPRester
    from pymatgen.ext.matproj import MPRestError
    mpr = MPRester(MAPI_KEY)
    a = []
    for i in df.mpids:
        try:
            a.append(mpr.query(i, properties =['theoretical' ] )[0]['theoretical'])
        except MPRestError:
            a.append('NA')
            print('MPRestError:',i)
    df['theoretical_only'] = a
    return df

# def add_mat_type(df):
    
#     """
#     Adds mat_type in the called df
#     """
    
#     mats_1 = [ 'As', 'Se' , 'Te', 'Br', 'Sb', 'Cl', 'Si']
#     mats_2 = ['O','S', 'C', 'N', 'P', 'I', 'F']
    
#     # def maybeMakeNumber(s):
#     #     """Returns a string 's' into a integer if possible, a float if needed or
#     #     returns it as is."""
    
#     #     # handle None, "", 0
#     #     if not s:
#     #         return s
#     #     try:
#     #         f = float(s)
#     #         i = int(f)
#     #         return i if f == i else f
#     #     except ValueError:
#     #         return s
    
#     def select_str(label):
#         """
#         Takes formula as input and returns a string removing numbers and brackets
#         """
#         a = []
#         for i in label:
#             try:
#                 int(i)
#             except ValueError:
#                 if i != '(' and i != ')':                
#                     a.append(i)
#         a = ''.join(a)
#         return a

#     for i in range(len(df)):
        
#         lab = select_str(df.loc[i]['formula'])  #Remove int and brackets from formula at i
#         df.at[i,'formula_noint'] = lab
#         for j in mats_1:
#             if (lab.find(j) != -1) and (len(lab) - lab.find(j) in [2,3,4]):    # 2 refers to last element, e.g. __As, 3 refers to second last, e.g. __As6
#                 df.at[i,'mat_type'] = j
            
#             else:
#                 #lab = list(df.loc[i]['formula'])
#                 #lab = list(map(maybeMakeNumber, lab)) #Converts numbers in lab as str to int
                
#                 for k in mats_2:
#                     try:
#                         if (lab[-1] == k) or (lab[-2] == k and type(lab[-1]) is int):    
#                             df.at[i,'mat_type'] = k
                        
#                     except IndexError:
#                         df.at[i,'mat_type'] = 'Others'
    
#         # Change nan to 'Others'   
#     for i in range(len(df)):
#         if pd.isnull(df.loc[i,'mat_type']):
#             df.at[i,'mat_type'] = 'Others'
                
#     return df

def add_anions(df):
    
    """
    Adds mat_type in the called df
    """
    
    anions = [ 'As', 'Se' , 'Te', 'Br', 'Sb', 'Cl', 'Si', 'O','S', 'C', 'N', 'P', 'I', 'F', 'H']
    
    
    def select_str(label):
        """
        Takes formula as input and returns a string removing numbers and brackets
        """
        a = []
        for i in label:
            try:
                int(i)
            except ValueError:
                if i != '(' and i != ')':                
                    a.append(i)
        a = ''.join(a)
        return a

    for i in range(len(df)):
        
        lab = select_str(df.loc[i]['formula'])  #Remove int and brackets from formula at i
        df.at[i,'formula_noint'] = lab
        lab = re.findall('[A-Z][^A-Z]*', lab)
        df.at[i,'no_of_mats'] = len(lab)
             
        for j in anions:
            if j == lab[-1]:
                if (len(lab) >= 3) and (lab[-2] in anions):
                    df.at[i,'mat_type'] = 'Double anions'
                else:
                    df.at[i,'mat_type'] = j
                break  # This improves accuracy and saves memory   
            else:
                try:
                      if (lab[-2] == j) and (len(lab) in [2,3,4,5]): # Only for quatarnaties and pantarnaries
                          df.at[i,'mat_type'] = j             
                except IndexError:   
                    df.at[i,'mat_type'] = 'Others'
    
        # Change nan to 'Others'   
    for i in range(len(df)):
        if pd.isnull(df.loc[i,'mat_type']):
            df.at[i,'mat_type'] = 'Others'
            
    def replace_names(data_frame):
        data_frame['mat_type'].replace(to_replace=['O'], value='Oxides',inplace=True)
        data_frame['mat_type'].replace(to_replace=['P'], value='Phosphides',inplace=True)
        data_frame['mat_type'].replace(to_replace=['S'], value='Sulphides',inplace=True)
        data_frame['mat_type'].replace(to_replace=['N'], value='Nitrides',inplace=True)
        data_frame['mat_type'].replace(to_replace=['As'], value='Arsenides',inplace=True)
        data_frame['mat_type'].replace(to_replace=['Se'], value='Selenides',inplace=True)
        data_frame['mat_type'].replace(to_replace=['Te'], value='Tellurides',inplace=True)
        data_frame['mat_type'].replace(to_replace=['Br'], value='Bromides',inplace=True)
        data_frame['mat_type'].replace(to_replace=['Sb'], value='Antimonides',inplace=True)
        data_frame['mat_type'].replace(to_replace=['Cl'], value='Chlorides',inplace=True)
        data_frame['mat_type'].replace(to_replace=['Si'], value='Silicides',inplace=True)
        data_frame['mat_type'].replace(to_replace=['C'], value='Carbides',inplace=True)
        data_frame['mat_type'].replace(to_replace=['I'], value='Iodides',inplace=True)
        data_frame['mat_type'].replace(to_replace=['F'], value='Fluorides',inplace=True)
        data_frame['mat_type'].replace(to_replace=['H'], value='Halides',inplace=True)
        return

    replace_names(df)
    
    return df 


def add_anions_v2(df):
    
    """
    Adds mat_type in the called df
    """
    
    anions = [ 'As', 'Se' , 'Te', 'Br', 'Sb', 'Cl', 'Si', 'O','S', 'C', 'N', 'P', 'I', 'F', 'H']
    
    
    def select_str(label):
        """
        Takes formula as input and returns a string removing numbers and brackets
        """
        a = []
        for i in label:
            try:
                int(i)
            except ValueError:
                if i != '(' and i != ')':                
                    a.append(i)
        a = ''.join(a)
        return a

    for i in range(len(df)):
        
        lab = select_str(df.loc[i]['formula'])  #Remove int and brackets from formula at i
        df.at[i,'formula_noint'] = lab
        lab = re.findall('[A-Z][^A-Z]*', lab)
        df.at[i,'no_of_mats'] = len(lab)
             
        for j in anions:
            if j == lab[-1]:
                if (len(lab) >= 3) and (lab[-2] in anions):
                    df.at[i,'mat_type'] = 'Double anions'
                else:
                    df.at[i,'mat_type'] = j
                break  # This improves accuracy and saves memory   
            else:
                try:
                      if (lab[-2] == j) and (len(lab) in [2,3,4,5]): # Only for quatarnaties and pantarnaries
                          df.at[i,'mat_type'] = j             
                except IndexError:   
                    df.at[i,'mat_type'] = 'Others'
    
        # Change nan to 'Others'   
    for i in range(len(df)):
        if pd.isnull(df.loc[i,'mat_type']):
            df.at[i,'mat_type'] = 'Others'
            
    def replace_names(data_frame):
        data_frame['mat_type'].replace(to_replace=['O'], value='Oxides',inplace=True)
        data_frame['mat_type'].replace(to_replace=['P'], value='Phosphides',inplace=True)
        data_frame['mat_type'].replace(to_replace=['S'], value='Chalcogenides',inplace=True)
        data_frame['mat_type'].replace(to_replace=['N'], value='Nitrides',inplace=True)
        data_frame['mat_type'].replace(to_replace=['As'], value='Arsenides',inplace=True)
        data_frame['mat_type'].replace(to_replace=['Se'], value='Chalcogenides',inplace=True)
        data_frame['mat_type'].replace(to_replace=['Te'], value='Chalcogenides',inplace=True)
        data_frame['mat_type'].replace(to_replace=['Br'], value='Halides',inplace=True)
        data_frame['mat_type'].replace(to_replace=['Sb'], value='Antimonides',inplace=True)
        data_frame['mat_type'].replace(to_replace=['Cl'], value='Halides',inplace=True)
        data_frame['mat_type'].replace(to_replace=['Si'], value='Silicides',inplace=True)
        data_frame['mat_type'].replace(to_replace=['C'], value='Carbides',inplace=True)
        data_frame['mat_type'].replace(to_replace=['I'], value='Halides',inplace=True)
        data_frame['mat_type'].replace(to_replace=['F'], value='Halides',inplace=True)
        data_frame['mat_type'].replace(to_replace=['H'], value='Hydrides',inplace=True)
        return

    replace_names(df)
    
    return df 


def add_cation(df):
    mats_1 = [ 'Li','Na','Rb','Cs','Be','Mg','Ca','Se','Ba','Sc','Lu','Ti','Zr','Hf','Nb','Ta','Cr','Mo','Mn','Tc','Re','Fe','Ru','Os','Co','Rh','Ir','Ni','Pd','Pt','Cu','Ag','Au','Zn','Cd','Hg','Al','Ga','In','Tl','Si','Ge','Sn','Pb','Sb','Bi','Te']
    mats_2 = ['K','Y', 'V', 'W', 'B', 'C']
    return df


