
import numpy as np
# import transforms_debug2



'''
    Module to create the training folder for the source-sep task containing the (X,Y) spectrograms

    Params:
    
        Wav_folder
        seq_dur
        FE_params
        target_source
        Fs
        other preproc

    TO DO:
        MULTIPROC
        parser args
'''


def padding_or_not(x,M,flag):
    #FUNCTION to PADD or TRUNCATE a signal x in order 
    #for it's length to be perfectly divisible by M ( L=k*M i.e. L is an integer multiple of M) 

    L = len(x)
    reminder = int(L%M)

    if reminder:
        #L%M!=0 (not perfect division)

        if flag:
            #ZERO PADDING CASE:

            #Find the number of zeros that we will pad
            nb_zeros = int( M*np.ceil(L/M) - L )
            x_new = np.concatenate(( x , np.zeros(nb_zeros) ))

        else:
            #TRUNCATING SAMPLES CASE:
            trunc_until = int(L - reminder)
            x_new = x[:trunc_until] 

    return x_new


def consec_segments_tensor(x,valid_seq_dur,sample_rate):
    #convert a full song to consecutive segments of seq_dur
    #(,nb_frames)-> (nb_samples,nb_frames)

    #if the song is longer than 4mins -> truncate it to 4 mins
    # if (x.shape[-1]/sample_rate)/60>=8:
    #     x = x[:,:,:sample_rate*180]

    #padding x in order for the lentgh of the full song to be an integer multiple of seq_dur length
    x = padding_or_not(x,valid_seq_dur*sample_rate,1) + np.finfo(np.float64).eps   #Adding epsilon in order to use STFT_mine :)

    nb_samples = int(len(x)/(valid_seq_dur*sample_rate)) 
    nb_frames = int(valid_seq_dur*sample_rate) 

    #reshaping to nb_segments,nb_frames,nb_channels and then permuting to get the correct tensor for the 
    x_segs = x.reshape(nb_samples , nb_frames)


    return x_segs 



def cputime():
    utime, stime, cutime, cstime, elapsed_time = os.times()
    return utime






#DETRMINING FE FORWARD-BACKWARD-----------------------------------------------------------------------------------------------

# //NSGT_SCALE_FRAMES
# "{ front_end_name : NSGT_SCALE_FRAMES , onset_det : custom , ksi_s : 44100 , min_scl : 512 , ovrlp_fact : 0.7 , middle_window : sg.hanning , matrix_form : 1 , multiproc : 1 }",


# //STFT_custom
# //"{ front_end_name : STFT_custom , a : 1024 , M : 4096 , support : 4096 }"

# //STFT_scipy
# //"{ front_end_name : scipy , a : 1024 , M : 4096 , support : 4096 }"

# //NSGT_CQT
# //"{ front_end_name : NSGT_CQT , ksi_s : 44100 , ksi_min : 32.07 , ksi_max : 10000 , B : 12 , matrix_form : 1 }"


# def get_Transform(params_dict,sig_len,dev="cpu"):

#     # need to globally configure an NSGT object to peek at its params to set up the neural network
#     # e.g. M depends on the sllen which depends on fscale+fmin+fmax

#     if dev:
#         device = dev
#     else:
#         use_cuda = torch.cuda.is_available()
#         device = torch.device("cuda" if use_cuda else "cpu")

#     nsgt_base = transforms_debug2.NSGTBase(
#         scale=params_dict["scl"],
#         fbins=params_dict["B"],
#         # N = ksi_s*int(args.seq_dur),
#         N = sig_len,
#         fmin=params_dict["ksi_min"],
#         fmax=params_dict["ksi_max"],
#         matrixform = params_dict["matrix_form"],
        
#         fs=Fs,
#         device = device,
#     )

#     nsgt, insgt = transforms_debug2.make_nsgt_filterbanks( 
#         nsgt_base  ,sample_rate=Fs
#     )
#     #cnorm = transforms.ComplexNorm(mono=nb_channels == 1)

#     forward = nsgt.to(device)
#     backward = insgt.to(device)
#     #cnorm = cnorm.to(device)

#     return forward , backward


def pick_front_end(front_end_params,seq_dur,Fs):  

    '''
        A FUNCTION to get front_end forward and backward methods---

            front_end_params : A dict containing the front end params
            seq_dur : SEQ-DUR IN SECONDS
            Fs : sampling rate
    '''
    


    #FRONT_ENDs available (its scalable)
    from Time_Frequency_Analysis.NSGT_CQT import NSGT_cqt
    from Time_Frequency_Analysis.SCALE_FRAMES import scale_frame
    from Time_Frequency_Analysis.STFT_custom import STFT_CUSTOM
    import nsgt as nsg      
    import scipy   
    import librosa  
    import numpy as np 
    
    
    front_end_lookup ={
        "STFT_custom":STFT_CUSTOM,
        "librosa":librosa,
        "scipy":scipy.signal,
        "nsgt_grr":nsg,
        "NSGT_CQT":NSGT_cqt,
        "NSGT_SCALE_FRAMES":scale_frame
    }


    #The STFTs and CQTs are L (signal len) dependend ie the only thing needed to construct the transform windows is L
    L = seq_dur*Fs
    #scale_frame is SIGNAL DEPENDEND i.e. in order to determine the windows positions (and consecuently construct them) you need the onsets
    #of the particular signal (its more complicated for the stereo chanel case so we test the mono)
    # mono_mix = librosa.to_mono(musdb_track.audio.T)


    front_end = front_end_lookup[front_end_params["front_end_name"]]

    if front_end==scipy.signal:
        g = np.hanning(front_end_params["support"])
        forward = lambda y : scipy.signal.stft( y , window=g, nfft=front_end_params["M"] , noverlap=front_end_params["a"] ,nperseg=front_end_params["support"])[-1]
        backward = lambda Y : scipy.signal.istft( Y, window=g, nfft=front_end_params["M"] , noverlap=front_end_params["a"]  ,nperseg=front_end_params["support"])[1]

    elif front_end==librosa:
        g = np.hanning(front_end_params["support"])
        #X = np.array( list( map( lambda x :  librosa.stft(x,n_fft=front_end_params["M"],hop_length=front_end_params["a"],win_length=front_end_params["support"],window=g) , track.audio.T ) ) )
        forward = lambda y : librosa.stft( y=y , n_fft=front_end_params["M"],hop_length=front_end_params["a"],win_length=front_end_params["support"],window=g ) 
        backward = lambda Y : librosa.istft( stft_matrix=Y ,hop_length=front_end_params["a"]  ,win_length=front_end_params["support"],window=g )
    
    elif front_end==STFT_CUSTOM:
        g = np.hanning(front_end_params["support"])
        stft = front_end(g,front_end_params["a"],front_end_params["M"],front_end_params["support"],L)
        forward = stft.forward  
        backward = stft.backward
    
    elif front_end==nsg:
        scale = nsg.LogScale
        scl = scale(front_end_params["ksi_min"], front_end_params["ksi_max"], front_end_params["B"] )
        nsgt = nsg.NSGT(scl, Fs, Ls=L, real=1, matrixform=1, reducedform=0 ,multithreading=0)
        forward = nsgt.forward
        backward = nsgt.backward
        # get_Transform(front_end_params,L)
        


    elif front_end==NSGT_cqt:
        nsgt = front_end(ksi_s=Fs,ksi_min=front_end_params["ksi_min"], ksi_max=front_end_params["ksi_max"], B=front_end_params["B"],L=L,matrix_form=1)
        forward = nsgt.forward
        backward = nsgt.backward


    


    # elif front_end==scale_frame:

    #     if front_end_params["onset_det"]=="custom":

    #         #Onset det custom using hpss to estimate the drums:
    #         D = librosa.stft(mono_mix)
    #         H, P = librosa.decompose.hpss(D, margin=(1.0,7.0))
    #         y_perc = librosa.istft(P)
    #         onsets = librosa.onset.onset_detect(y=y_perc, sr=Fs, units="samples")

    #     else:
    #         onsets = librosa.onset.onset_detect(y=mono_mix, sr=Fs, units="samples")


    #     middle_window = np.hanning if front_end_params["middle_window"]=="np.hanning" else scipy.signal.tukey
        
    #     scl_frame_object = front_end(ksi_s=Fs,min_scl=front_end_params["min_scl"],overlap_factor=front_end_params["ovrlp_fact"],onset_seq=onsets,middle_window=middle_window,L=L,matrix_form=front_end_params["matrix_form"],multiproc=front_end_params["multiproc"])
    #     forward = scl_frame_object.forward
    #     backward = scl_frame_object.backward




    return forward , backward





# def create_spec_pair_list(dir_pth):
#     for count,it in enumerate(os.scandir(dir_pth)):
        
#         #LOADING------------------------------------------
#         (x,_),(y,_) = librosa.load( it.path + '/mixture.wav' , sr = Fs , mono= True ) , librosa.load( it.path + '/vocals.wav' , sr = Fs , mono= True )



#         #DOWNSAMPLING------------------------------------------



#         #SEGMENTING------------------------------------------
#         x_segs , y_segs = consec_segments_tensor(x,seq_dur,Fs)   , consec_segments_tensor(y,seq_dur,Fs)  
        



#         #FORWARD------------------------------------------
        
#         #maybe use dB
#         Forward = lambda x : np.abs( librosa.stft( x , n_fft=nfft, hop_length=nhop , win_length=nfft ) )

#         if count:
#             Spec_seg_pair_list = Spec_seg_pair_list + random.sample( list( map( lambda seg_x_y : ( torch.from_numpy( np.array([Forward(seg_x_y[0])]) ).float() , torch.from_numpy( np.array([Forward(seg_x_y[1])]) ).float() ) , zip(x_segs,y_segs) ) ) , len(x_segs) )
#         else:
#             Spec_seg_pair_list = random.sample( list( map( lambda seg_x_y : ( torch.from_numpy( np.array([Forward(seg_x_y[0])]) ).float() , torch.from_numpy( np.array([Forward(seg_x_y[1])]) ).float()  )  , zip(x_segs,y_segs) ) ) , len(x_segs) ) 


#     return Spec_seg_pair_list

#IMPORTANT FUNC
def get_Forward_segs_from_one_song(x,dataset_params,Forward):
    #Segment-Forward-Preproc---------------------------------------

    # import multiprocessing
    # pool = multiprocessing.Pool(processes=6)

    #Segment
    x_segs  = consec_segments_tensor(x,dataset_params["seq_dur"],dataset_params["Fs"])   

    #Forward
    # pool.imap( lambda seg_x_y : ( torch.from_numpy( np.array([Forward(seg_x_y[0])]) ).float() , torch.from_numpy( np.array([Forward(seg_x_y[1])]) ).float() ) , zip(x_segs,y_segs) )
    # X_segs = pool.map( Forward  , x_segs ) 
    X_segs = list( map( Forward  , x_segs ) )

    #Preproc (i.e. dB , standardization)


    return X_segs


def create_spec_pair_list(root_train_dir):

    
    # pbar = tqdm.tqdm( enumerate(os.scandir(root_train_dir)) )
    pbar = tqdm.tqdm( os.scandir(root_train_dir) )

    for count,it in enumerate(pbar):  

        pbar.set_description("Training batch")

        


        #LOADING - segmenting - Forward------------------------------------------
        (x,_),(y,_) = librosa.load( it.path + '/mixture.wav' , sr = Fs , mono= True ) , librosa.load( it.path + '/'+target_source+'.wav' , sr = Fs , mono= True )


        X_segs , Y_segs =  get_Forward_segs_from_one_song(x,dataset_params,Forward) , get_Forward_segs_from_one_song(y,dataset_params,Forward)

        #Convert to spectrograms (amplitude) and maybe Preproc----------------------------------------------
        X_segs_amps , Y_segs_amps = list( map( lambda seg : np.abs(seg) , X_segs ) ) , list( map( lambda seg : np.abs(seg) , Y_segs ) ) 

        # XY_seg_amps =  list( map( lambda seg : (np.abs(seg[0]),np.abs(seg[1])) , zip(X_segs,Y_segs) ) ) 



        #MERGING THE TWO LISTS [(X_seg,Y_seg),..] and converting to torch tensors
        if count:
            Spec_seg_pair_list = Spec_seg_pair_list + random.sample( list( map( lambda seg : ( torch.from_numpy( np.array([seg[0]]) ).float() , torch.from_numpy( np.array([seg[1]]) ).float() ) , zip(X_segs_amps,Y_segs_amps) ) ) , len(X_segs) )
        else:
            Spec_seg_pair_list = random.sample( list( map( lambda seg : ( torch.from_numpy( np.array([seg[0]]) ).float() , torch.from_numpy( np.array([seg[1]]) ).float() ) , zip(X_segs_amps,Y_segs_amps) ) ) , len(X_segs) ) 


    return Spec_seg_pair_list


if __name__ == "__main__":

    import numpy as np
    import argparse
    import librosa
    import random
    import torch
    import os
    import yaml    
    import json
    import tqdm
    from pathlib import Path


    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset-params', '--Dataset-params', type=yaml.safe_load,
                            help='provide Transform parameters as a quoted json sting')

    # parser.add_argument('-front-end-params', '--FE-params', type=yaml.safe_load,
    #                         help='provide Transform parameters as a quoted json sting')
                            


    args = parser.parse_args()    


    dataset_params = args.Dataset_params 
    Wav_folder = args.Dataset_params["Wav_folder"]
    Target_folder = args.Dataset_params["Target_folder"]
    Fs = args.Dataset_params["Fs"]
    target_source = args.Dataset_params["target_source"]
    seq_dur = args.Dataset_params["seq_dur"]
    FE_params = args.Dataset_params["FE_params"]
    Forward , _ = pick_front_end(front_end_params = FE_params, seq_dur=seq_dur, Fs=Fs)

    root_train_dir = Wav_folder+'/train/train'
    root_valid_dir = Wav_folder+'/valid'    

    #CREATE TARGET FOLDER
    target_path = Path(Target_folder)
    target_path.mkdir(parents=True, exist_ok=True)    

    #CREATE spec pair list-------------------------------------------------------------------------------
    t1 = cputime()

    Spec_seg_pair_list_train = create_spec_pair_list(root_train_dir)         

    Spec_seg_pair_list_valid = create_spec_pair_list(root_valid_dir)

    #Saving--------------------------------------------------------------------------------------------
    torch.save(Spec_seg_pair_list_train, Target_folder+'/Spec_seg_pair_list_train.pt')

    torch.save(Spec_seg_pair_list_valid, Target_folder+'/Spec_seg_pair_list_valid.pt')

    t2 = cputime()

    Calc_saving_Time = t2 - t1

    #Creating a json containing all the neccesary DATASET params and then Saving---------------------------
    dataset_params["Calc_Saving_Time"] = str(Calc_saving_Time/60)+" mins"

    json_object = json.dumps(dataset_params, indent=4)
    
    # Writing 
    with open(Target_folder+"/Dataset_Params_log.json", "w") as outfile:
        outfile.write(json_object)    

