def plot_main(df_ds1, df_ds2, df_ds3, save):
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    order = [ 'Oxides','Sulphides', 'Nitrides', 'Phosphides', 'Arsenides', 'Selenides' , 'Tellurides', 'Bromides', 'Antimonides', 'Chlorides', 'Silicides', 'Carbides', 'Iodides', 'Fluorides', 'Halides', 'Others','Double anions']
    fig = plt.figure()
    ax = fig.add_subplot(3, 2, 1)
    g1 = sns.countplot(data = df_ds1, x='mat_type', ax=ax, order = order)
    g1.set(xlabel=None, xticklabels=[], ylabel = 'Count', yscale="log",ylim = (1,1000000))
    ax = fig.add_subplot(3, 2, 2)
    g2 = sns.histplot(data=df_ds1, x='bg_mp', ax=ax, kde = True)
    g2.set(xlabel=None, ylabel = None)
    ax = fig.add_subplot(3, 2, 3)
    g3 = sns.countplot(data = df_ds2, x='mat_type', ax=ax, order = order)
    g3.set(xlabel=None, xticklabels=[],ylabel = 'Count')
    ax = fig.add_subplot(3, 2, 4)
    g4 = sns.histplot(data=df_ds2, x='bg_exp', ax=ax, kde = True)
    g4.set(xlabel=None, ylabel = None)
    ax = fig.add_subplot(3, 2, 5)
    g5 = sns.countplot(data = df_ds3, x='mat_type', ax=ax, order = order)
    g5.set(xlabel=None,ylabel = 'Count', ylim = (0,8), yticks = ([0,3,6]))
    
    plt.xticks(rotation=90)
    
    ax = fig.add_subplot(3, 2, 6)
    g6 = sns.histplot(data=df_ds3, x='bg_exp', ax=ax, kde = True)
    g6.set(xlabel='Band gaps (eV)', ylabel = None)
    
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.3, wspace=0.2)
    if save:
        plt.savefig('plots/Dist_main.jpeg', dpi=300)  
    plt.show()
    if save:
        plt.close('all')
    return


def plot_main_v2(df_ds1, df_ds2, df_ds3, save):
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    order = [ 'Oxides','Chalcogenides', 'Nitrides', 'Phosphides', 'Arsenides', 'Halides', 'Antimonides', 'Silicides', 'Carbides', 'Hydrides', 'Others','Double anions']
    fig = plt.figure()
    ax = fig.add_subplot(3, 2, 1)
    g1 = sns.countplot(data = df_ds1, x='mat_type', ax=ax, order = order)
    g1.set(xlabel=None, xticklabels=[], ylabel = 'Count', yscale="log",ylim = (1,1000000))
    ax = fig.add_subplot(3, 2, 2)
    g2 = sns.histplot(data=df_ds1, x='bg_mp', ax=ax, kde = True)
    g2.set(xlabel=None, ylabel = None)
    ax = fig.add_subplot(3, 2, 3)
    g3 = sns.countplot(data = df_ds2, x='mat_type', ax=ax, order = order)
    g3.set(xlabel=None, xticklabels=[],ylabel = 'Count')
    ax = fig.add_subplot(3, 2, 4)
    g4 = sns.histplot(data=df_ds2, x='bg_exp', ax=ax, kde = True)
    g4.set(xlabel=None, ylabel = None)
    ax = fig.add_subplot(3, 2, 5)
    g5 = sns.countplot(data = df_ds3, x='mat_type', ax=ax, order = order)
    g5.set(xlabel=None,ylabel = 'Count', ylim = (0,8), yticks = ([0,3,6]))
    
    plt.xticks(rotation=90)
    
    ax = fig.add_subplot(3, 2, 6)
    g6 = sns.histplot(data=df_ds3, x='bg_exp', ax=ax, kde = True)
    g6.set(xlabel='Band gaps (eV)', ylabel = None)
    
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.3, wspace=0.2)
    if save:
        plt.savefig('plots/Dist_main.jpeg', dpi=300)  
    plt.show()
    if save:
        plt.close('all')
    return

def plot_no_of_elements(df_ds1, df_ds2, df_ds3, save):
    
    def annotate():
        for p in ax.patches:
                x=p.get_bbox().get_points()[:,0]
                y=p.get_bbox().get_points()[1,1]
                if np.isnan(y):
                    y = 0
                ax.annotate(int(y), (x.mean(), y), ha='center', va = 'bottom',fontsize=8) 
        return       
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    order = [ 1,2, 3, 4, 5, 6 , 7, 8]
    fig = plt.figure()
    ax = fig.add_subplot(3, 1, 1)
    g1 = sns.countplot(data = df_ds1, x='no_of_mats', order = order, ax=ax)
    g1.set(xlabel=None, xticklabels=[], ylabel = 'Count', ylim = (0,11000))
    annotate()
    ax = fig.add_subplot(3, 1, 2)
    g2 = sns.countplot(data = df_ds2, x='no_of_mats', ax=ax, order = order)
    g2.set(xlabel=None, xticklabels=[],ylabel = 'Count', ylim = (0,270))
    annotate()
    ax = fig.add_subplot(3, 1, 3)
    g3 = sns.countplot(data = df_ds3, x='no_of_mats', ax=ax, order = order)
    g3.set(xlabel='No. of constituent elements',ylabel = 'Count', ylim = (0,26))
    annotate()
    plt.xticks(fontsize=12,weight = 'bold')
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.2)
    if save:
        plt.savefig('plots/Dist_no_of_elements.jpeg', dpi=300)  
    plt.show()
    if save:
        plt.close('all')
    return