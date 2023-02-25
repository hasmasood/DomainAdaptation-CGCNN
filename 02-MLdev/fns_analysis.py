# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 11:47:11 2022

@author: z5022637
"""

def bg_master_copy(bg_exp, bg_mp, bg_hse, bg_gw, bg_m1, bg_m2, ax_symbol):
    
    import pandas as pd
    import seaborn as sb
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    from sklearn import metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
         
    #Write metrics
    
    mae_mp = round(mean_absolute_error(bg_exp , bg_mp),2)
    mae_hse = round(mean_absolute_error(bg_exp , bg_hse),2)
    mae_gw = round(mean_absolute_error(bg_exp , bg_gw),2)
    mae_m1 = round(mean_absolute_error(bg_exp , bg_m1),2)
    mae_m2 = round(mean_absolute_error(bg_exp , bg_m2),2)
    
    rmse_mp = round(mean_squared_error(bg_exp , bg_mp, squared=False),2) #squared=False will give RMSE, true with give MSE
    rmse_hse = round(mean_squared_error(bg_exp , bg_hse, squared=False),2)
    rmse_gw = round(mean_squared_error(bg_exp , bg_gw, squared=False),2)
    rmse_m1 = round(mean_squared_error(bg_exp , bg_m1, squared=False),2)
    rmse_m2 = round(mean_squared_error(bg_exp , bg_m2, squared=False),2)
    
    r2_mp = round(r2_score(bg_exp , bg_mp),2)
    r2_hse = round(r2_score(bg_exp , bg_hse),2)
    r2_gw = round(r2_score(bg_exp , bg_gw),2)
    r2_m1 = round(r2_score(bg_exp , bg_m1),2)
    r2_m2 = round(r2_score(bg_exp , bg_m2),2)
    

    file_met = open('Metrics.txt', 'w')
    file_met.write('MAE_mp = {}, RMSE_mp = {}, R2_mp= {}\n'.format(mae_mp, rmse_mp, r2_mp))
    file_met.write('MAE_hse = {}, RMSE_hse = {}, R2_hse= {}\n'.format(mae_hse, rmse_hse, r2_hse))
    file_met.write('MAE_gw = {}, RMSE_gw = {}, R2_gw= {}\n'.format(mae_gw, rmse_gw, r2_gw))
    file_met.write('MAE_m1 = {}, RMSE_m1 = {}, R2_m1= {}\n'.format(mae_m1, rmse_m1, r2_m1))
    file_met.write('MAE_m2 = {}, RMSE_m2 = {}, R2_m2= {}\n'.format(mae_m2, rmse_m2, r2_m2))
    file_met.close()
    
    df_metrics = pd.DataFrame({'MP':[mae_mp, rmse_mp, r2_mp],
                            'HSE':[mae_hse, rmse_hse, r2_hse],
                            'GW': [mae_gw, rmse_gw, r2_gw],
                            'M1': [mae_m1, rmse_m1, r2_m1],
                            'M2': [mae_m2, rmse_m2, r2_m2],
                            'Index': ['MAE (eV)','RMSE (eV)','R2']
        })
    df_metrics.set_index('Index',inplace=True)
    df_metrics.to_excel('Metrics.xlsx')
    
    
    #Plot mp
    df_mp = pd.DataFrame(bg_exp.values, columns = ['Actual'])
    df_mp['Predict'] = bg_mp
    df_mp['Error'] = bg_mp-bg_exp
    
    sbp = sb.jointplot(
        x='Actual',
        y='Predict',
        data=df_mp,
        color = 'b',
        kind='scatter' # or 'kde' or 'hex'    
        
    )
    
    xmin, xmax = sbp.ax_joint.get_xlim()
    ymin, ymax = sbp.ax_joint.get_ylim()
    lims = [min(xmin, ymin), max(xmax, ymax)]
    sbp.ax_joint.plot(lims, lims,'-k' ,alpha=0.75, zorder=0,linewidth=2, transform=sbp.ax_joint.transData)
    
    sbp.ax_joint.set(ylim=lims)
    sbp.ax_joint.set(xlim=lims)
    
    sbp.ax_joint.set(xlabel="Experimental values ("+ax_symbol+")", ylabel="Predicted values ("+ax_symbol+")")
    
    sbp.ax_joint.text(xmin, ymax,"$R^2$ = {}\nMAE = {} {}".format(r2_mp,mae_mp,ax_symbol), fontsize=12)  
    
    plt.title('PBE')
    plt.tight_layout()
    #sbp.savefig('plot_mp.jpeg', dpi=300)  
    plt.show()
    plt.close('all')
    

    #Plot m1
    df_m1 = pd.DataFrame(bg_exp.values, columns = ['Actual'])
    df_m1['Predict'] = bg_m1
    df_m1['Error'] = bg_m1-bg_exp
    
    sbp = sb.jointplot(
        x='Actual',
        y='Predict',
        data=df_m1,
        color = 'r',
        kind='scatter' # or 'kde' or 'hex'    
        
    )
    
    xmin, xmax = sbp.ax_joint.get_xlim()
    ymin, ymax = sbp.ax_joint.get_ylim()
    lims = [min(xmin, ymin), max(xmax, ymax)]
    sbp.ax_joint.plot(lims, lims,'-k' ,alpha=0.75, zorder=0,linewidth=2, transform=sbp.ax_joint.transData)
    
    sbp.ax_joint.set(ylim=lims)
    sbp.ax_joint.set(xlim=lims)
    
    sbp.ax_joint.set(xlabel="Experimental values ("+ax_symbol+")", ylabel="Predicted values ("+ax_symbol+")")
    
    sbp.ax_joint.text(xmin, ymax,"$R^2$ = {}\nMAE = {} {}".format(r2_m1,mae_m1,ax_symbol), fontsize=12)  
    
    plt.title('M1')
    plt.tight_layout()
    #sbp.savefig('plot_m1.jpeg', dpi=300)  
    plt.show()
    plt.close('all')
    
 

    #Plot m2
    
    df_m2 = pd.DataFrame(bg_exp.values, columns = ['Actual'])
    df_m2['Predict'] = bg_m2
    df_m2['Error'] = bg_m2-bg_exp
    
    sbp = sb.jointplot(
        x='Actual',
        y='Predict',
        data=df_m2,
        color = 'g',
        kind='scatter' # or 'kde' or 'hex'    
        
    )
    
    xmin, xmax = sbp.ax_joint.get_xlim()
    ymin, ymax = sbp.ax_joint.get_ylim()
    lims = [min(xmin, ymin), max(xmax, ymax)]
    sbp.ax_joint.plot(lims, lims,'-k' ,alpha=0.75, zorder=0,linewidth=2, transform=sbp.ax_joint.transData)
    
    sbp.ax_joint.set(ylim=lims)
    sbp.ax_joint.set(xlim=lims)
    
    sbp.ax_joint.set(xlabel="Experimental values ("+ax_symbol+")", ylabel="Predicted values ("+ax_symbol+")")
    
    sbp.ax_joint.text(xmin, ymax,"$R^2$ = {}\nMAE = {} {}".format(r2_m2,mae_m2,ax_symbol), fontsize=12)  
    
    plt.title('M2')
    plt.tight_layout()
    #sbp.savefig('plot_m2.jpeg', dpi=300)  
    plt.show()
    plt.close('all')



    #Plot hse
    
    df_hse = pd.DataFrame(bg_exp.values, columns = ['Actual'])
    df_hse['Predict'] = bg_hse
    df_hse['Error'] = bg_hse-bg_exp
    
    sbp = sb.jointplot(
        x='Actual',
        y='Predict',
        data=df_hse,
        color = 'brown',
        kind='scatter' # or 'kde' or 'hex'    
        
    )
    
    xmin, xmax = sbp.ax_joint.get_xlim()
    ymin, ymax = sbp.ax_joint.get_ylim()
    lims = [min(xmin, ymin), max(xmax, ymax)]
    sbp.ax_joint.plot(lims, lims,'-k' ,alpha=0.75, zorder=0,linewidth=2, transform=sbp.ax_joint.transData)
    
    sbp.ax_joint.set(ylim=lims)
    sbp.ax_joint.set(xlim=lims)
    
    sbp.ax_joint.set(xlabel="Experimental values ("+ax_symbol+")", ylabel="Predicted values ("+ax_symbol+")")
    
    sbp.ax_joint.text(xmin, ymax,"$R^2$ = {}\nMAE = {} {}".format(r2_hse,mae_hse,ax_symbol), fontsize=12)  
    
    plt.title('HSE')
    plt.tight_layout()
    #sbp.savefig('plot_hse.jpeg', dpi=300)  
    plt.show()
    plt.close('all')

    #Plot gw
    
    df_gw = pd.DataFrame(bg_exp.values, columns = ['Actual'])
    df_gw['Predict'] = bg_gw
    df_gw['Error'] = bg_gw-bg_exp
    
    sbp = sb.jointplot(
        x='Actual',
        y='Predict',
        data=df_gw,
        color = 'orange',
        kind='scatter' # or 'kde' or 'hex'    
        
    )
    
    xmin, xmax = sbp.ax_joint.get_xlim()
    ymin, ymax = sbp.ax_joint.get_ylim()
    lims = [min(xmin, ymin), max(xmax, ymax)]
    sbp.ax_joint.plot(lims, lims,'-k' ,alpha=0.75, zorder=0,linewidth=2, transform=sbp.ax_joint.transData)
    
    sbp.ax_joint.set(ylim=lims)
    sbp.ax_joint.set(xlim=lims)
    
    sbp.ax_joint.set(xlabel="Experimental values ("+ax_symbol+")", ylabel="Predicted values ("+ax_symbol+")")
    
    sbp.ax_joint.text(xmin, ymax,"$R^2$ = {}\nMAE = {} {}".format(r2_gw,mae_gw,ax_symbol), fontsize=12)  
    
    plt.title('GW')
    plt.tight_layout()
    #sbp.savefig('plot_gw.jpeg', dpi=300)  
    plt.show()
    plt.close('all')

    #Combined plot M1+M2
    concatenated = pd.concat([df_m1.assign(dataset='df_m1'), df_m2.assign(dataset='df_m2')]).reset_index(drop=True)

    sbp = sb.jointplot(data=concatenated, x="Actual", y="Predict", hue="dataset", kind = 'scatter',
                        palette=dict(df_m1="r", df_m2="g"),
                        alpha=0.6                   
                        )
    handles, labels = sbp.ax_joint.get_legend_handles_labels()
    sbp.ax_joint.legend(handles=handles, labels=['M1 (MAE = {} {})'.format(mae_m1,ax_symbol), 'M2 (MAE = {} {})'.format(mae_m2,ax_symbol)],
                      title="Model")
    
    xmin, xmax = sbp.ax_joint.get_xlim()
    ymin, ymax = sbp.ax_joint.get_ylim()
    lims = [min(xmin, ymin), max(xmax, ymax)]
    sbp.ax_joint.plot(lims, lims,'-k' ,alpha=0.75, zorder=0,linewidth=2, transform=sbp.ax_joint.transData)
    
    sbp.ax_joint.set(ylim=lims)
    sbp.ax_joint.set(xlim=lims)
    
    sbp.ax_joint.set(xlabel="Experimental values ("+ax_symbol+")", ylabel="Predicted values ("+ax_symbol+")")
    
    plt.tight_layout()
    #sbp.savefig('plot_m1_m2.jpeg', dpi=300)  
    plt.show()
    plt.close('all')
    
    # # Plotting errors
    import numpy as np
    from scipy import stats


    #Plot errors - mp
    stats_mp = {'mean': round(np.mean(bg_mp-bg_exp),2),
      'median': round(np.median(bg_mp-bg_exp),2),
      'mode': round(stats.mode(bg_mp-bg_exp)[0][0],2),
      'skew': round(stats.skew(bg_mp-bg_exp),2)
      } 
    
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(111)
    sbp = sb.histplot(ax=ax1,data=df_mp, x='Error', kde=True, color = 'b')
    plt.xlabel("Error ŷ-y ({})".format(ax_symbol), fontsize = 12)
    plt.ylabel("Counts", fontsize = 12)   
    ax1.text(0.02, 0.98, "Mean = {} {}\nMedian = {} {}\nMode = {} {}\nSkewness = {}".format(
        stats_mp['mean'],ax_symbol,stats_mp['median'],ax_symbol,  stats_mp['mode'],ax_symbol,stats_mp['skew']
                                                                                           ), ha="left", va="top", transform=ax1.transAxes,fontsize=12)
    plt.title('PBE')
    plt.tight_layout()
    #plt.savefig('Error_mp.jpeg', dpi=300)  
    plt.show()
    plt.close('all')


    #Plot errors - m1

    stats_m1 = {'mean': round(np.mean(bg_m1-bg_exp),2),
      'median': round(np.median(bg_m1-bg_exp),2),
      'mode': round(stats.mode(bg_m1-bg_exp)[0][0],2),
      'skew': round(stats.skew(bg_m1-bg_exp),2)
      } 
    
    
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(111)
    sbp = sb.histplot(ax=ax1,data=df_m1, x='Error', kde=True, color = 'r')
    plt.xlabel("Error ŷ-y ({})".format(ax_symbol), fontsize = 12)
    plt.ylabel("Counts", fontsize = 12)   
    ax1.text(0.02, 0.98, "Mean = {} {}\nMedian = {} {}\nMode = {} {}\nSkewness = {}".format(
        stats_m1['mean'],ax_symbol,stats_m1['median'],ax_symbol,  stats_m1['mode'],ax_symbol,stats_m1['skew']
                                                                                           ), ha="left", va="top", transform=ax1.transAxes,fontsize=12)
    plt.title('M1')
    plt.tight_layout()
    #plt.savefig('Error_m1.jpeg', dpi=300)  
    plt.show()
    plt.close('all')


    #Plot errors - m2

    stats_m2 = {'mean': round(np.mean(bg_m2-bg_exp),2),
      'median': round(np.median(bg_m2-bg_exp),2),
      'mode': round(stats.mode(bg_m2-bg_exp)[0][0],2),
      'skew': round(stats.skew(bg_m2-bg_exp),2)
      } 
    
    
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(111)
    sbp = sb.histplot(ax=ax1,data=df_m2, x='Error', kde=True, color = 'g')
    plt.xlabel("Error ŷ-y ({})".format(ax_symbol), fontsize = 12)
    plt.ylabel("Counts", fontsize = 12)   
    ax1.text(0.02, 0.98, "Mean = {} {}\nMedian = {} {}\nMode = {} {}\nSkewness = {}".format(
        stats_m2['mean'],ax_symbol,stats_m2['median'],ax_symbol,  stats_m2['mode'],ax_symbol,stats_m2['skew']
                                                                                           ), ha="left", va="top", transform=ax1.transAxes,fontsize=12)
    plt.title('M2')
    plt.tight_layout()
    #plt.savefig('Error_m2.jpeg', dpi=300)  
    plt.show()
    plt.close('all')    


    #Plot errors - hse
    stats_hse = {'mean': round(np.mean(bg_hse-bg_exp),2),
      'median': round(np.median(bg_hse-bg_exp),2),
      'mode': round(stats.mode(bg_hse-bg_exp)[0][0],2),
      'skew': round(stats.skew(bg_hse-bg_exp),2)
      } 
    

    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(111)
    sbp = sb.histplot(ax=ax1,data=df_hse, x='Error', kde=True, color = 'brown')
    plt.xlabel("Error ŷ-y ({})".format(ax_symbol), fontsize = 12)
    plt.ylabel("Counts", fontsize = 12)   
    ax1.text(0.02, 0.98, "Mean = {} {}\nMedian = {} {}\nMode = {} {}\nSkewness = {}".format(
        stats_hse['mean'],ax_symbol,stats_hse['median'],ax_symbol,  stats_hse['mode'],ax_symbol,stats_hse['skew']
                                                                                           ), ha="left", va="top", transform=ax1.transAxes,fontsize=12)
    plt.title('HSE')
    plt.tight_layout()
    #plt.savefig('Error_hse.jpeg', dpi=300)  
    plt.show()
    plt.close('all')

    #Plot errors - gw
    stats_gw = {'mean': round(np.mean(bg_gw-bg_exp),2),
      'median': round(np.median(bg_gw-bg_exp),2),
      'mode': round(stats.mode(bg_gw-bg_exp)[0][0],2),
      'skew': round(stats.skew(bg_gw-bg_exp),2)
      } 
    
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(111)
    sbp = sb.histplot(ax=ax1,data=df_gw, x='Error', kde=True, color = 'orange')
    plt.xlabel("Error ŷ-y ({})".format(ax_symbol), fontsize = 12)
    plt.ylabel("Counts", fontsize = 12)   
    ax1.text(0.02, 0.98, "Mean = {} {}\nMedian = {} {}\nMode = {} {}\nSkewness = {}".format(
        stats_gw['mean'],ax_symbol,stats_gw['median'],ax_symbol,  stats_gw['mode'],ax_symbol,stats_gw['skew']
                                                                                           ), ha="left", va="top", transform=ax1.transAxes,fontsize=12)
    plt.title('GW')
    plt.tight_layout()
    #plt.savefig('Error_gw.jpeg', dpi=300)  
    plt.show()
    plt.close('all')


    #Plot error histogram - Combined
    sbp = sb.displot(data=concatenated, x='Error', kde=True, hue = 'dataset', 
                      palette=dict(df_m1="r", df_m2="g"), 
                      common_norm=False
                      ) #No axis argument in displot
    new_labels = ['M1 (Mean = {} {})'.format(stats_m1['mean'], ax_symbol), 'M2 (Mean = {} {})'.format(stats_m2['mean'], ax_symbol)]
    for t, l in zip(sbp._legend.texts, new_labels): t.set_text(l)
    sbp._legend.set_title('Models')
    plt.xlabel("Error ŷ-y ({})".format(ax_symbol), fontsize = 12)
    plt.ylabel("Counts", fontsize = 12)
    plt.tight_layout()
    #plt.savefig('Error_m1_m2.jpeg', dpi=300)  
    plt.show()
    plt.close('all')

    return
