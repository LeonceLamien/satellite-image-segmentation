# from Networks.model import *
# from Dataset.makeGraph import *

# import argparse
# import yaml

# import os
# from os.path import dirname, abspath
# from termcolor import colored


# rootDirectory    = dirname(abspath(__file__))
# datasetDirectory = os.path.join(rootDirectory,    "Dataset")
# imgDirectory     = os.path.join(datasetDirectory, "images")
# maskDirectory    = os.path.join(datasetDirectory, "annotations")

# parser = argparse.ArgumentParser()
# parser.add_argument('-exp', type=str, default='DefaultExp')


# ######################################################################################
# #
# # MAIN PROCEDURE 
# # launches an experiment whose parameters are described in a yaml file  
# # 
# # Example of use in the terminal: python main.py -exp DefaultExp
# # with 'DefaultExp' beeing the name of the yaml file (in the Todo_list folder) with 
# # the wanted configuration 
# # 
# ######################################################################################

# def main(parser):
#     # -----------------
#     # 0. INITIALISATION 
#     # -----------------
#     # Read the yaml configuration file 
#     stream = open('Todo_List/' + parser.exp + '.yaml', 'r')
#     param  = yaml.safe_load(stream)
#     # Path to the folder that will contain results of the experiment 
#     resultsPath = os.path.join(rootDirectory, "Results", parser.exp)

#     # ------------------------
#     # 1. NETWORK INSTANTIATION 
#     # ------------------------
#     myNetwork  = Network_Class(param, imgDirectory, maskDirectory, resultsPath)

#     # ------------------------------------------ 
#     # 2. VISUALISATION OF THE DATASET (OPTIONAL)
#     # ------------------------------------------
#     # Comment line below to skip the visualisation
#     showDataset(myNetwork.dataSetTrain, param)

#     # ------------------
#     # 3. TRAIN THE MODEL  
#     # ------------------
#     print(colored('Start to train the network', 'red'))
#     myNetwork.train()
#     print(colored('The network is trained', 'red'))
    
#     # ---------------------
#     # 4. EVALUATE THE MODEL  
#     # ---------------------  
#     myNetwork.loadWeights()
#     myNetwork.evaluate()
    

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-exp', type=str, default='ColorizationExp')
#     args = parser.parse_args()
    
#     # Exécuter l'expérience de colorisation
#     main(args)


from Networks.model import *
from Dataset.makeGraph import *

import argparse
import yaml

import os
from os.path import dirname, abspath
from termcolor import colored


rootDirectory    = dirname(abspath(__file__))
datasetDirectory = os.path.join(rootDirectory,    "Dataset")
imgDirectory     = os.path.join(datasetDirectory, "images")
maskDirectory    = os.path.join(datasetDirectory, "annotations")

parser = argparse.ArgumentParser()
parser.add_argument('-exp', type=str, default='DefaultExp')
parser.add_argument('--skip-clustering', action='store_true', help='Skip clustering analysis')


######################################################################################
#
# MAIN PROCEDURE 
# launches an experiment whose parameters are described in a yaml file  
# 
# Example of use in the terminal: python main.py -exp DefaultExp
# with 'DefaultExp' beeing the name of the yaml file (in the Todo_list folder) with 
# the wanted configuration 
# 
######################################################################################

def main(parser):
    # -----------------
    # 0. INITIALISATION 
    # -----------------
    # Read the yaml configuration file 
    stream = open('Todo_List/' + parser.exp + '.yaml', 'r')
    param  = yaml.safe_load(stream)
    # Path to the folder that will contain results of the experiment 
    resultsPath = os.path.join(rootDirectory, "Results", parser.exp)

    # ------------------------
    # 1. NETWORK INSTANTIATION 
    # ------------------------
    myNetwork  = Network_Class(param, imgDirectory, maskDirectory, resultsPath)

    # ------------------------------------------ 
    # 2. VISUALISATION OF THE DATASET (OPTIONAL)
    # ------------------------------------------
    # Comment line below to skip the visualisation
    showDataset(myNetwork.dataSetTrain, param)

    # ------------------
    # 3. TRAIN THE MODEL  
    # ------------------
    print(colored('Start to train the network', 'red'))
    myNetwork.train()
    print(colored('The network is trained', 'red'))
    
    # ---------------------
    # 4. EVALUATE THE MODEL  
    # ---------------------  
    myNetwork.loadWeights()
    myNetwork.evaluate()
    
    # -------------------------
    # 5. CLUSTERING ANALYSIS
    # -------------------------
    if not parser.skip_clustering and param["MODEL"].get("TASK", "segmentation") == "colorization":
        print(colored('\n🔍 Performing clustering analysis on learned features', 'blue'))
        # Le clustering est déjà effectué dans la méthode train() pour la colorisation
        print(colored('Clustering analysis completed!', 'blue'))
        
        # Afficher un résumé des résultats
        clustering_path = os.path.join(resultsPath, "Clustering")
        if os.path.exists(clustering_path):
            print(f"\n📊 Clustering results saved in: {clustering_path}")
            print("Contains:")
            print("  - cluster_analysis.txt: Detailed analysis of clusters")
            print("  - clusters_tsne.png: 2D visualization of clusters")
            print("  - test_cluster_analysis.txt: Analysis on test set")
            print("  - pseudo_segmentation_evaluation.txt: Quality metrics")
            print("  - Visualizations/: Example visualizations")
            print("  - Pseudo_Segmentation_Maps/: Generated segmentation maps")
    

if __name__ == '__main__':
    parser = parser.parse_args()
    main(parser)
