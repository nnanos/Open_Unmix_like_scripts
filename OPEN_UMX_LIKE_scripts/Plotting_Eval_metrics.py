import pandas as pd
import numpy as np
import seaborn as sns

def agg_metrics_over_frames(Eval_Log_json,scores_per_track):
    #AGGRAGATE METRICS FOR EACH TRACK OVER FRAMES-----------------------------------------------------------------------------------------

    # if args.eval_mthd_params["eval_mthd"] == "BSS_eval_mus_track":
    #     pass
    # else:

    aggregation_method_lookup = {
        "median":np.median,
        "mean":np.mean
    }        
    aggregation_method_func = aggregation_method_lookup[Eval_Log_json["eval_mthd_params"]["aggregation_method"]]

    
    Testing_tracks_list_dirs = list( map(lambda dir :  dir.path , os.scandir(Eval_Log_json["root_TEST_dir"]) ) )
    nb_tracks = len(Testing_tracks_list_dirs)

    track_scores_agg_over_frames = []



    for k in range(nb_tracks):

        tmp_agg_scores = aggregation_method_func(scores_per_track[k],0)

        #track_scores_df_tmp = pd.DataFrame(data= tmp_agg_scores , index=targets , columns = ["SDR","ISR","SIR","SAR"]  ) 
        
        track_scores_agg_over_frames.append( tmp_agg_scores )  

    return track_scores_agg_over_frames



def agg_metrics_over_tracks(Eval_Log_json,track_scores_agg_over_frames):
    #AGGRAGATE METRICS OVER TRACKS-----------------------------------------------------------------------------------------
    
    sources_targets = [ Eval_Log_json["target"] , "residual" ]
    
    # if args.eval_mthd_params["eval_mthd"] == "BSS_eval_mus_track":
    #     results = museval.EvalStore( frames_agg=args.eval_mthd_params["aggregation_method"], tracks_agg=args.eval_mthd_params["aggregation_method"] )
    #     for k in range(len(Testing_tracks_list_dirs)):
    #         results.add_track(scores_per_track[k])

    # else:

    aggregation_method_lookup = {
        "median":np.median,
        "mean":np.mean
    }        
    aggregation_method_func = aggregation_method_lookup[Eval_Log_json["eval_mthd_params"]["aggregation_method"]]


    metrics_agg_over_frames_tracks = aggregation_method_func(np.array(track_scores_agg_over_frames),0) 

    if (Eval_Log_json["eval_mthd_params"]["eval_mthd"] == "BSSeval_custom") or Eval_Log_json["eval_mthd_params"]["eval_mthd"] == "mir_eval":
        metrics_agg_over_frames_tracks = pd.DataFrame(data=metrics_agg_over_frames_tracks , index=sources_targets , columns = ["SDR","SIR","SAR"]  )  

    # elif Eval_Log_json["eval_mthd_params"]["eval_mthd"] == "mir_eval":
    #     metrics_agg_over_frames_tracks = metrics_agg_over_frames_tracks[:,:metrics_agg_over_frames_tracks.shape[1]-1]
    #     # metrics_agg_over_frames_tracks = pd.DataFrame(data=metrics_agg_over_frames_tracks , index=sources_targets , columns = ["SDR","ISR","SIR","SAR"]  )            
    #     metrics_agg_over_frames_tracks = pd.DataFrame(data=metrics_agg_over_frames_tracks , index=sources_targets , columns = ["SDR","SIR","SAR"]  )
    else:
        metrics_agg_over_frames_tracks = metrics_agg_over_frames_tracks[:,:metrics_agg_over_frames_tracks.shape[1]-1]
        # metrics_agg_over_frames_tracks = pd.DataFrame(data=metrics_agg_over_frames_tracks , index=sources_targets , columns = ["SDR","ISR","SIR","SAR"]  )
        metrics_agg_over_frames_tracks = pd.DataFrame(data=metrics_agg_over_frames_tracks , index=sources_targets , columns = ["SDR","SIR","SAR"]  )            


    return metrics_agg_over_frames_tracks


def get_dict(path_list):

    methods_dict = {}    


    for path in path_list:
        #LOADING the Eval_Log--------------------------------------------------
        with open(path+"/Eval_Log.json", 'r') as openfile:
            # Reading from json file
            Eval_Log_json = json.load(openfile)     


        method_name = Eval_Log_json["method_name"]    
        target_source = Eval_Log_json["target"]


        #LOADING the scores_per_frames_per_track.pickle --------------------------------------------------
        with open(path+'/scores_per_frames_per_track.pickle', 'rb') as handle:
            scores_per_frames_per_track = pickle.load(handle)    


        # #Agg metrics_over_frames to create the BOXPLOT
        track_scores_agg_over_frames = agg_metrics_over_frames(Eval_Log_json,scores_per_frames_per_track)
        track_scores_agg_over_frames = np.array(track_scores_agg_over_frames)

        Testing_tracks_list_dirs = list( map(lambda dir :  dir.path , os.scandir(Eval_Log_json["root_TEST_dir"]) ) )
        nb_tracks = len(Testing_tracks_list_dirs)    
        tracks_l = list( map(lambda i : "track"+str(i) , np.arange(nb_tracks)) )
        tmp_dict = {
            target_source : pd.DataFrame(data=track_scores_agg_over_frames[:,0,:] , index=tracks_l , columns = ["SDR","SIR","SAR"]  ) ,
            "residual": pd.DataFrame(data=track_scores_agg_over_frames[:,1,:] , index=tracks_l , columns = ["SDR","SIR","SAR"]  )
        }   
        

        #Agg metrics_agg_over_frames_tracks to see the BIG PICTURE
        metrics_agg_over_frames_tracks_DF = agg_metrics_over_tracks(Eval_Log_json,track_scores_agg_over_frames)
        # print(metrics_agg_over_frames_tracks)    


        methods_dict[method_name] = [ tmp_dict , track_scores_agg_over_frames , metrics_agg_over_frames_tracks_DF ]


    return methods_dict


def aux(methods_dict,path_list,target_source):

    df_dict={}
    for path in path_list:
        #LOADING the Eval_Log--------------------------------------------------
        with open(path+"/Eval_Log.json", 'r') as openfile:
            # Reading from json file
            Eval_Log_json = json.load(openfile)     


        method_name = Eval_Log_json["method_name"]    
        # target_source = Eval_Log_json["target"]

        methods_dict[method_name][0]

        df_dict[method_name] = methods_dict[method_name][0][target_source]


    return df_dict



def create_boxplot_row_one_source(df_dict,sort_by,source):
    #create_boxplot_row for one target source for multiple methods

    #list of dfs , each element represanting a method of shape (track,metric)

    method_names = list(df_dict.keys())

    df_list = list(df_dict.values())
    
    tracks_l = np.arange(len(df_list[0].index))

    #SDR
    methods_per_SDR = np.array(list( map(lambda df_meth : df_meth["SDR"] , df_list) ))
    df_methods_per_SDR = pd.DataFrame(data=methods_per_SDR.T , index=tracks_l , columns = method_names  )

    #SIR
    methods_per_SIR = np.array(list( map(lambda df_meth : df_meth["SIR"] , df_list) ))
    df_methods_per_SIR = pd.DataFrame(data=methods_per_SIR.T , index=tracks_l , columns = method_names  )


    #SAR
    methods_per_SAR = np.array(list( map(lambda df_meth : df_meth["SAR"] , df_list) ))
    df_methods_per_SAR = pd.DataFrame(data=methods_per_SAR.T , index=tracks_l , columns = method_names  )



    fig, axs = plt.subplots(ncols=3)
    sns.boxplot( data=df_methods_per_SDR, ax=axs[0] , orient="h")
    sns.boxplot( data=df_methods_per_SIR, ax=axs[1] , orient="h")
    sns.boxplot( data=df_methods_per_SAR, ax=axs[2] , orient="h")

    fig.suptitle("Boxplots for target source="+source+" for all the methods used", fontsize=14)
    fig.tight_layout()


def create_boxplots(path_list):
    
    
    #CREATE BOXPLOTS------------------------------------------------------------------------------------
    methods_dict = get_dict(path_list)
    
    
    df_dict = aux(methods_dict,path_list,target_source="vocals") 
    
    #create one row for vocals
    create_boxplot_row_one_source(df_dict=df_dict,sort_by=1,source="vocals")


    df_dict = aux(methods_dict,path_list,target_source="residual") 
    
    #create one row for residual
    create_boxplot_row_one_source(df_dict=df_dict,sort_by=1,source="residual")    

    plt.show()

if __name__ == "__main__":

    import json
    import numpy as np
    import argparse
    import yaml
    import tqdm
    import os
    import pickle
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()



    parser.add_argument(
        "--evaldirs",
        nargs="+",
        default=["vocals", "drums", "bass", "other"],
        type=str,
        help="provide evaldirs to be processed. \
              If none, all available evaldirs will be computed",
    )

    args = parser.parse_args()   

    path_list = args.evaldirs



    #CREATE BOXPLOTS------------------------------------------------------------------------------------
    # create_boxplots( path_list)


    #Get the bigger picture by printing the aggregated over frames and over tracks metrics for each method
    methods_dict = get_dict(path_list)

    #Target source:
    for key in methods_dict.keys():
        print("\n\n\n\nThe metrics aggregated over frames and over tracks for method "+key+" are:\n")
        print(methods_dict[key][2])
        print("\n\n\n\n")


a = 0    