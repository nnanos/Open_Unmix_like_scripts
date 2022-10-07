import model
from pathlib import Path
import torch
import librosa
import numpy as np



#Load model------
def load_target_models(target, model_path="umxhq", device="cpu", pretrained=True):
    """
        INPUTS:
            target : the target source that your model is trained on
            model_path : the dir where the target.pth is located
            device : the device on which the processing (model computations) will be done

    """

    #Load the state of the model---------------------
    target_model_path = model_path+"/"+target+".pth"
    state = torch.load(target_model_path, map_location=device)    

    #INITIALIZE THE MODEL------------------
    
    #UNET arcitecture
    unmix = model.UNet().to(device)           

    unmix.load_state_dict(state, strict=False)

    
    return unmix




def Separate(X_segs,model,Backward):
    '''
        INPUTs:
            x : mono mixture
            seq_dur : sequence duration in seconds that we feed into the NN
            Fs : sampling rate 
    '''
    

    X_amp = np.abs(X_segs)

    X_phase = np.angle(X_segs)


    with torch.no_grad():
        X = torch.from_numpy(X_amp).float()
        target_spectrogram = model(X)


    X_angle = torch.from_numpy(X_phase).float()
    real = target_spectrogram * torch.cos(X_angle)
    imag = target_spectrogram * torch.sin(X_angle)  
    target_stft_segs = torch.complex(real, imag).squeeze().cpu().detach().numpy()  


    x_segs = np.array( list( map( lambda seg_x_y :   Backward(seg_x_y)    , target_stft_segs )    )   )     

    x_out = x_segs.flatten()

    return x_out


def pick_eval_mthd_and_eval_track(eval_mthd_params,refrences,targets,estimates_ndarray):

    #FUNCTION TO PRODUCE BSS_EVAL METRICS FOR ONE TRACK
    #Inputs:
    #   eval_mthd_params: a dict containig the bss_eval method params
    #   musdb_track: 
    #   estimates_dict:
    #Outputs:
    #   

    import museval
    import mir_eval
    import BSSeval_custom

    eval_mthd_lookup ={
        "BSS_eval_mus_track":museval.eval_mus_track,
        "BSS_evaluation":museval.evaluate,
        #"mir_eval":mir_eval.separation.bss_eval_sources,
        "mir_eval":mir_eval.separation.bss_eval_sources_framewise,
        "BSSeval_custom":BSSeval_custom.evaluation

    }        

    eval_mthd = eval_mthd_lookup[eval_mthd_params["eval_mthd"]] 



    # if eval_mthd_params["eval_mthd"] == "BSS_eval_mus_track":
    #     results = museval.EvalStore( frames_agg=eval_mthd_params["aggregation_method"], tracks_agg=eval_mthd_params["aggregation_method"] )

    #EVALUATION---------------------------------------------------------------------------------------------------------------------------------------
    if eval_mthd_params["nb_chan"] == 2 :
        #STEREO eval methods------------------------------------------------------------------------------------------------------------------

        if eval_mthd_params["eval_mthd"] == "BSS_eval_mus_track":
            #EVALUATE METH1 BSS_eval (eval_mus_track)--------------------------------------------------------------------
            #results.add_track(eval_mthd(track, estimates,hop=eval_mthd_params["hop"],win=eval_mthd_params["win"]))   
            track_scores = eval_mthd(musdb_track, estimates_dict,hop=eval_mthd_params["hop"],win=eval_mthd_params["win"])
          


        if eval_mthd_params["eval_mthd"] == "BSS_evaluation":
            #EVALUATE METH2  BSS_eval (evaluate)--------------------------------------------------------------------------
            # targets = list(musdb_track.sources.keys())
            refrences = np.array(list( map( lambda x : musdb_track.sources[x].audio , targets) ))        
            estimates_ndarray = np.array(list( map( lambda x : estimates_dict[x] , targets) ))
            tmp = eval_mthd(refrences,estimates_ndarray,hop=eval_mthd_params["hop"],win=eval_mthd_params["win"] )
            track_scores = np.array(tmp).T 
              

        if eval_mthd_params["eval_mthd"] == "mir_eval":
            #EVALUATE METH3  BSS_eval (mir_eval.separation.bss_eval_images_framewise)--------------------------------------------------------------------------
            # targets = list(musdb_track.sources.keys())
            refrences = np.array(list( map( lambda x : musdb_track.sources[x].audio , targets) ))        
            estimates_ndarray = np.array(list( map( lambda x : estimates_dict[x] , targets) ))
            tmp = eval_mthd(refrences,estimates_ndarray,hop=eval_mthd_params["hop"],window=eval_mthd_params["win"] ,compute_permutation=False )
            track_scores = np.array(tmp).T 


        # if eval_mthd_params["eval_mthd"] == "BSSeval_custom":
        #     #EVALUATE METH3  BSS_eval (mir_eval.separation.bss_eval_images_framewise)--------------------------------------------------------------------------
        #     # targets = list(musdb_track.sources.keys())
        #     refrences = np.array(list( map( lambda x : musdb_track.sources[x].audio , targets) ))        
        #     estimates_ndarray = np.array(list( map( lambda x : estimates_dict[x] , targets) ))
        #     tmp = eval_mthd(refrences,estimates_ndarray,hop=eval_mthd_params["hop"],window=eval_mthd_params["win"] ,compute_permutation=False )
        #     track_scores = np.array(tmp).T                  
                          


    else:
        #WE PERFORM THE SEPARATION IN THE SINGLE chanel mix (mono) 

        #MONO eval methods------------------------------------------------------------------------------------------------------------------

        # if args.est_mthd_params["est_mthd_name"] == HPSS:
        #     refrences = np.concatenate(([np.array(musdb_track.sources["vocals"].audio+musdb_track.sources["bass"].audio+musdb_track.sources["other"].audio)],[musdb_track.sources["drums"].audio]))    
        # else:    
        #     refrences = np.array(list( map( lambda x : musdb_track.sources[x].audio , targets) ))

        # refrences = np.array(list( map( lambda x : musdb_track.sources[x].audio , targets) ))

        
        # estimates_ndarray = np.array(list( map( lambda x : estimates_dict[x] , targets) ))    
        # #converting the references to mono      
        # refrences = np.array(list(map(lambda x : librosa.to_mono(x.T) , refrences)))      
        # #estimates_ndarray = np.array(list(map(lambda x : librosa.to_mono(x.T) , estimates_ndarray)))

        # #ADDING THE LAST (nb_chanels) DIMENSION
        # refrences = np.array([refrences.T]).T
        # estimates_ndarray = np.array([estimates_ndarray.T]).T                          

        if eval_mthd_params["eval_mthd"] == "BSS_evaluation":
            #EVALUATE METH2  BSS_eval (evaluate)--------------------------------------------------------------------------
            sdr,_,sir,sar = eval_mthd(refrences,estimates_ndarray,hop=eval_mthd_params["hop"],win=eval_mthd_params["win"])
            track_scores = np.array([sdr,sir,sar]).T 
                

        
        if eval_mthd_params["eval_mthd"] == "mir_eval":
            #EVALUATE METH3 mir----------------------------------------------  

            estimates_ndarray = estimates_ndarray[:,:,0]   
            tmp = eval_mthd(refrences, estimates_ndarray, hop=eval_mthd_params["hop"],window=eval_mthd_params["win"],compute_permutation=False )[:-1]
            track_scores = np.array(tmp).T 
              


        if eval_mthd_params["eval_mthd"] == "BSSeval_custom":
            #EVALUATE METH4----------------------------------------------  

            #REMOVING THE LAST (nb_chanels) DIMENSION
            # refrences = refrences[:,:,0]
            estimates_ndarray = estimates_ndarray[:,:,0]   

            # track_scores = np.array(  list(map( lambda tmp : eval_mthd(tmp[0],musdb_track,tmp[1],win=eval_mthd_params["win"]) , zip(estimates_ndarray,targets) ))  )
            track_scores_target = eval_mthd(estimates_ndarray,refrences,win=eval_mthd_params["win"])
            track_scores_residual = eval_mthd(np.flip(estimates_ndarray,axis=0),np.flip(refrences,axis=0),win=eval_mthd_params["win"])

            track_scores = np.array([ track_scores_target , track_scores_residual ]) 




            if len(track_scores.shape)==3:
                #THEN number of sources to estimate is greater than 1 
                track_scores = track_scores.transpose(1,0,2)
            else:
                track_scores = np.array([track_scores])



    return track_scores    




def seperate_and_evaluate(test_track_path_dir):
    
    #SEPERATE_AND_EVALUATE-----------------------------------------------------------------------------------------
    if Dataset_params["target_source"] == "vocals":
        # residual = "accompaniment"
        
        (x,_) , (y,_) , (res,_) = librosa.load( test_track_path_dir+"/mixture.wav" , sr = Dataset_params["Fs"] , mono= True ) , librosa.load( test_track_path_dir+"/"+Dataset_params["target_source"]+".wav" , sr = Dataset_params["Fs"] , mono= True ) , librosa.load( test_track_path_dir+"/accompaniment.wav" , sr = Dataset_params["Fs"] , mono= True )   
    else:
        (x,_) , (y,_) , (res,_) = librosa.load( test_track_path_dir+"/mixture.wav" , sr = Dataset_params["Fs"] , mono= True ) , librosa.load( test_track_path_dir+"/"+Dataset_params["target_source"]+".wav" , sr = Dataset_params["Fs"] , mono= True ) , librosa.load( test_track_path_dir+"/linear_mixture.wav" , sr = Dataset_params["Fs"] , mono= True )   
        res = res - y



    #SEGMENTING and FORWARDING------------------------------------------
    X_segs = get_Forward_segs_from_one_song(x,Dataset_params,Forward)
    X_segs = np.expand_dims(np.array(X_segs),1)
    x_hat = Separate(X_segs,unmix,Backward)

    #Truncating to the length of the samples of the mix (because we have zeropadded in order to segment the signal)
    if len(x)<len(x_hat):
        x_hat = x_hat[:len(x)]
    else:
        x = x[:len(x_hat)]
        y = y[:len(x_hat)]
        res = res[:len(x_hat)]
    #---------------------------------

    res_hat = x - x_hat

    estimates_dict = {
        Dataset_params["target_source"] : x_hat,
        "residual" : res_hat
    }


    estimates_ndarray = np.expand_dims( np.concatenate( ( np.array([estimates_dict[Dataset_params["target_source"]]]) , np.array([estimates_dict["residual"]]) )  )  , axis = 2 )
    references = np.concatenate( ( np.array([ y ]) , np.array([ res ]) )  )


    sources_targets = list(estimates_dict.keys())




    #Performing evaluation-----------------------------------------------
    scores = pick_eval_mthd_and_eval_track(eval_mthd_params=args.eval_mthd_params,refrences=references,targets=sources_targets,estimates_ndarray=estimates_ndarray) 


    return scores


def cputime():
    utime, stime, cutime, cstime, elapsed_time = os.times()
    return utime

if __name__ == "__main__":

    from Audio_proc_lib.audio_proc_functions import load_music,sound_write
    from Data import get_Forward_segs_from_one_song , pick_front_end
    import json
    import numpy as np
    import argparse
    import yaml
    import tqdm
    import os


    parser = argparse.ArgumentParser()

    parser.add_argument('--method-name', type=str, default="/home/nnanos/Desktop/Spectrograms_tst",
                            help='Flag to add the sisec18 methods for comparison  with the current method that we examine ')           

    parser.add_argument('--Model_dir', type=str, default="/home/nnanos/Desktop/Spectrograms_tst",
                            help='Flag to add the sisec18 methods for comparison  with the current method that we examine ')                            


    parser.add_argument('--root_TEST_dir', type=str, default="/home/nnanos/Desktop/Spectrograms_tst",
                            help='Flag to add the sisec18 methods for comparison  with the current method that we examine ')                            


    parser.add_argument('--evaldir', type=str, default="/home/nnanos/Desktop/Spectrograms_tst",
                            help='Flag to add the sisec18 methods for comparison  with the current method that we examine ')                            


    parser.add_argument('--target', type=str, default="/home/nnanos/Desktop/Spectrograms_tst",
                            help='Flag to add the sisec18 methods for comparison  with the current method that we examine ')                            


    parser.add_argument('--cores', type=int, default=1,
                            help='Flag to add the sisec18 methods for comparison  with the current method that we examine ')                            


    parser.add_argument('-eval-params', '--eval_mthd_params', type=yaml.safe_load,
                            help='provide evaluation method parameters as a quoted json sting')     

    args = parser.parse_args()   



    #LOADING MODEL AND FRONT_END--------------------------------------------------------------------------------
    model_path = args.Model_dir

    with open(model_path+'/'+args.target+'.json', 'r') as openfile:
        # Reading from json file
        train_json = json.load(openfile)    

    Dataset_params = train_json["args"]["Dataset_params"]
      

    #Load FE
    Forward , Backward = pick_front_end(front_end_params = Dataset_params["FE_params"], seq_dur= Dataset_params["seq_dur"], Fs = Dataset_params["Fs"] )

    #Load model
    unmix = load_target_models(target=Dataset_params["target_source"],model_path=model_path)




    #SEPERATE_AND_EVALUATE over the TEST dir-----------------------------------------------------------------------------------------
    # if Dataset_params["target_source"] == "vocals":
    #     residual = "accompaniment"

    sources_targets = [  Dataset_params["target_source"] , "residual" ]

    Testing_tracks_list_dirs = list( map(lambda dir :  dir.path , os.scandir(args.root_TEST_dir) ) )


    t1 = cputime()


    if args.cores > 1:

        
        import multiprocessing

        pool = multiprocessing.Pool(args.cores)
        # multiprocessing.set_start_method("spawn")


        scores_per_track = list(
            pool.imap_unordered(
                seperate_and_evaluate,
                iterable=Testing_tracks_list_dirs,
                chunksize=1,
            )
        )
        pool.close()
        pool.join()
        # for scores in scores_list:
        #     results.add_track(scores)

    else:
        scores_per_track = []
        for test_track_path_dir in tqdm.tqdm( Testing_tracks_list_dirs ):
            scores_per_track.append( seperate_and_evaluate(test_track_path_dir) )

            # print(test_track_path_dir, "\n", scores)
            # results.add_track(scores)    


    t2 = cputime()

    Calc_Time = t2 - t1



    # #AGGRAGATE METRICS FOR EACH TRACK OVER FRAMES-----------------------------------------------------------------------------------------
    # import pandas as pd
    
    # if args.eval_mthd_params["eval_mthd"] == "BSS_eval_mus_track":
    #     pass
    # else:

    #     aggregation_method_lookup = {
    #         "median":np.median,
    #         "mean":np.mean
    #     }        
    #     aggregation_method_func = aggregation_method_lookup[args.eval_mthd_params["aggregation_method"]]

    #     track_scores_agg_over_frames = []



    #     for k in range(len(Testing_tracks_list_dirs)):

    #         tmp_agg_scores = aggregation_method_func(scores_per_track[k],0)

    #         #track_scores_df_tmp = pd.DataFrame(data= tmp_agg_scores , index=targets , columns = ["SDR","ISR","SIR","SAR"]  ) 
            
    #         track_scores_agg_over_frames.append( tmp_agg_scores )  


    # #AGGRAGATE METRICS OVER TRACKS-----------------------------------------------------------------------------------------
    # if args.eval_mthd_params["eval_mthd"] == "BSS_eval_mus_track":
    #     results = museval.EvalStore( frames_agg=args.eval_mthd_params["aggregation_method"], tracks_agg=args.eval_mthd_params["aggregation_method"] )
    #     for k in range(len(Testing_tracks_list_dirs)):
    #         results.add_track(scores_per_track[k])

    # else:

    #     metrics_agg_over_frames_tracks = aggregation_method_func(np.array(track_scores_agg_over_frames),0) 

    #     if args.eval_mthd_params["eval_mthd"] == "BSSeval_custom":
    #         metrics_agg_over_frames_tracks = pd.DataFrame(data=metrics_agg_over_frames_tracks , index=sources_targets , columns = ["SDR","SIR","SAR"]  )  

    #     elif args.eval_mthd_params["eval_mthd"] == "mir_eval":
    #         metrics_agg_over_frames_tracks = metrics_agg_over_frames_tracks[:,:metrics_agg_over_frames_tracks.shape[1]-1]
    #         # metrics_agg_over_frames_tracks = pd.DataFrame(data=metrics_agg_over_frames_tracks , index=sources_targets , columns = ["SDR","ISR","SIR","SAR"]  )            
    #         metrics_agg_over_frames_tracks = pd.DataFrame(data=metrics_agg_over_frames_tracks , index=sources_targets , columns = ["SDR","SIR","SAR"]  )
    #     else:
    #         # metrics_agg_over_frames_tracks = pd.DataFrame(data=metrics_agg_over_frames_tracks , index=sources_targets , columns = ["SDR","ISR","SIR","SAR"]  )
    #         metrics_agg_over_frames_tracks = pd.DataFrame(data=metrics_agg_over_frames_tracks , index=sources_targets , columns = ["SDR","SIR","SAR"]  )            

    #     results = metrics_agg_over_frames_tracks


#     # sound_write(x_out,Dataset_params["Fs"],"/home/nnanos/tst.wav")


    #SAVING EVAL RESULTS AND LOGS to evaldir-------------------------------------------------
    # create evaldir dir if not exist
    eval_dir_path = Path(args.evaldir)
    eval_dir_path.mkdir(parents=True, exist_ok=True)


    #Saving logs-----------------------------------------------------
    Eval_Log  = vars(args)
    Eval_Log["method_name"]=args.method_name
    Eval_Log["Calculation_Sep_Eval_time"] = str(Calc_Time/60) + " mins"  
    json_object = json.dumps(Eval_Log, indent=4)
    
    # Writing 
    with open(args.evaldir+'/Eval_Log.json', "w") as outfile:
        outfile.write(json_object)    


    #Saving metrics---------------------------------------------
    import pickle 
    with open(args.evaldir+'/scores_per_frames_per_track.pickle', 'wb') as handle:
        pickle.dump(scores_per_track, handle, protocol=pickle.HIGHEST_PROTOCOL)    



a = 0