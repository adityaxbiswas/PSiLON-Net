



from os import path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, sharex=True, sharey=True, figsize=(4.8, 6.4))

dataset_name = 'higgs'

for model_name in ['standard', 'standard2', 'psilon']:
    identifier = dataset_name + ' ' + model_name
    project_folder = path.abspath(path.join(path.sep,"Projects", "PSiLON-Net"))
    data_folder = path.join(project_folder, "data", identifier+" net")

    lambda_dict = {
        "higgs psilon": ['1e-4', '2.5e-4', '5e-4', '1e-3', '5e-3'],
        "higgs standard": ['2.5e-3', '5e-3', '1e-2', '1.3e-2', '1.7e-2'],
        "higgs standard2": ['2.5e-3', '5e-3', '1e-2', '1.3e-2', '1.7e-2']
    }
    lambda_list = lambda_dict[identifier]
    lambda_list_pretty = ["λ="+val for val in lambda_list]
    lambda_list = ["lambda-"+val for val in lambda_list]
    data_graph = pd.read_csv(path.join(data_folder,
                                        lambda_list[0]+'.csv'))
    data_graph.drop("Wall time", axis = 1, inplace = True)
    data_graph.rename({"Value": lambda_list_pretty[0]}, axis = 1, inplace = True)
    for i in range(1,len(lambda_list)):
        data_graph[lambda_list_pretty[i]] = pd.read_csv(path.join(data_folder,
                                        lambda_list[i]+'.csv')).loc[:,"Value"]
        

    y_lim_dict = {
        "higgs psilon": [0.6,1.5],
        "higgs standard": [0.6,1.5],
        "higgs standard2": [0.6,1.5] 
    }
    ax_dict = {"higgs standard2": 0, "higgs standard": 1, "higgs psilon": 2}

    sns.lineplot(x='Step', y='value', hue='variable', 
                        data=pd.melt(data_graph, ['Step']), ax=axes[ax_dict[identifier]],
                        legend=None)
    axes[ax_dict[identifier]].set(ylim = y_lim_dict[identifier], 
                                  xlim = [0,5000],
                                  ylabel="Validation RMSE" if identifier == "higgs standard" else None,
                                  xlabel = "Optimization Step")
    axes[ax_dict[identifier]].yaxis.set_ticks(np.arange(0.6, 1.5, 0.1),
                                              labels = [0.6, '', 0.8, '', 1.0, '', 1.2, '', 1.4])
    axes[ax_dict[identifier]].yaxis.grid()
    axes[ax_dict[identifier]].tick_params(axis='both', which='major', labelsize=9)

plt.yticks(fontsize=8)
plt.xticks(fontsize=8)
plt.show()


fig.savefig(path.join(project_folder, "data", dataset_name+".png"),
             dpi=600, bbox_inches='tight')