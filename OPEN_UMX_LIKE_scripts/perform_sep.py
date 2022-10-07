import model
from pathlib import Path
import torch
import librosa
import numpy as np
from evaluate import load_target_models,Separate






if __name__ == "__main__":

    from Audio_proc_lib.audio_proc_functions import load_music,sound_write
    from Data import get_Forward_segs_from_one_song , pick_front_end
    import json
    import numpy as np
    import argparse



    parser = argparse.ArgumentParser()


    parser.add_argument('--Model_dir', type=str, default="/home/nnanos/Desktop/Spectrograms_tst",
                            help='Flag to add the sisec18 methods for comparison  with the current method that we examine ')   


    parser.add_argument('--input-wav', type=str, default="/home/nnanos/Desktop/Spectrograms_tst",
                            help='Flag to add the sisec18 methods for comparison  with the current method that we examine ') 


    parser.add_argument('--out_filename', type=str, default="/home/nnanos/Desktop/Spectrograms_tst",
                            help='Flag to add the sisec18 methods for comparison  with the current method that we examine ') 


    args = parser.parse_args()  


    model_path = args.Model_dir

    import fnmatch
    import os

    for file in os.listdir(model_path):
        if fnmatch.fnmatch(file, '*.json'):
            with open(model_path+'/'+file, 'r') as openfile:
                # Reading from json file
                train_json = json.load(openfile)  
  

    Dataset_params = train_json["args"]["Dataset_params"]   

    Forward , Backward = pick_front_end(front_end_params = Dataset_params["FE_params"], seq_dur= Dataset_params["seq_dur"], Fs = Dataset_params["Fs"] )


    #x,_ = librosa.load( args.input_wav , sr = Dataset_params["Fs"] , mono= True )
    x,s = load_music() 


    # x_resampled
    x = librosa.resample( x , orig_sr=s, target_sr=Dataset_params["Fs"] )

    #Load model
    unmix = load_target_models(target=Dataset_params["target_source"],model_path=model_path)


    #SEGMENTING and FORWARDING------------------------------------------
    X_segs = get_Forward_segs_from_one_song(x,Dataset_params,Forward)
    X_segs = np.expand_dims(np.array(X_segs),1)
    x_out = Separate(X_segs,unmix,Backward)

    if len(x)<len(x_out):
        x_out = x_out[:len(x)]
    else:
        x = x[:len(x_out)] 

    x_res = x - x_out



    # x_out = Separate(x,unmix,Dataset_params["seq_dur"],Dataset_params["Fs"])


    sound_write(x_out,Dataset_params["Fs"],args.out_filename)


a = 0