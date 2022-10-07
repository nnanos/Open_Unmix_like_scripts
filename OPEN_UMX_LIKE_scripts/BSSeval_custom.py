import numpy as np
import librosa
import museval
# from Audio_proc_lib.audio_proc_functions import  padding_or_not



#---------------------------------------------------------------------------------------------------------------------

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

def compute_energy_ratios(s_target,e_interf,e_artif):

    #computing the energy ratios
    # metrics = {}
    # metrics['SDR'] = 10*np.log10( np.dot( s_target,s_target ) / np.dot( (e_interf+e_artif) , (e_interf+e_artif) ) )
    # metrics['SIR'] = 10*np.log10( np.dot( s_target,s_target ) / np.dot( e_interf,e_interf ) )
    # metrics['SAR'] = 10*np.log10( np.dot( s_target + e_interf , s_target + e_interf ) / np.dot( e_artif,e_artif ) )

    return [10*np.log10( ( np.dot( s_target,s_target ) + np.finfo(np.float32).eps )/ (np.dot( (e_interf+e_artif) , (e_interf+e_artif) ) + np.finfo(np.float32).eps ) ), 
            10*np.log10(( np.dot( s_target,s_target ) + np.finfo(np.float32).eps )/( np.dot( e_interf,e_interf ) + np.finfo(np.float32).eps )),
            10*np.log10( (np.dot( s_target + e_interf , s_target + e_interf ) + np.finfo(np.float32).eps) / ( np.dot( e_artif,e_artif ) + np.finfo(np.float32).eps ) )
            ]  

def evaluation(estimates,refrences,orthogonality_assumption=False,win=None,padding_flag=None):
    #EVALUATION FUNCTION IMPLEMENTED BASED ON THE BSSeval metrics paper
    #estimate : mono signal estimate
    #musdb_track : the object that represents a Test track from the musdb dataset 
    #target : string options ['vocals','drums','bass','other']

    target_estimate = estimates[0]

    target_source = refrences[0]
    res_source = refrences[1]



    s_target = ( np.dot( target_source,target_estimate ) / np.dot(target_source,target_source) ) * target_source


    S = np.c_[target_source,res_source]

    if orthogonality_assumption:
        #obtainig the dot products of the target_estimate with all the sources (i.e. the geometric representation of the target_estimate in the basis defined by the sources)
        cords = np.dot(S.T,target_estimate)

        #obtaining the basis signals (normalizing each column by its energy)
        #S = np.c_[vocals/(np.linalg.norm(vocals)**2) , bass/(np.linalg.norm(bass)**2),other/(np.linalg.norm(other)**2),drums/(np.linalg.norm(drums)**2) ]
        #each row contains the sources normalized by its energy
        S = np.array(list(map(lambda s : s/np.dot(s,s), S.T)))

        #obtaining the projection of the estimate in the basis subspace
        Proj_to_S_subspace = np.dot(S.T,cords)

        #we can obtain the e_interf by subtraction because we assume an orthogonal basis (the source signals are mutualy orthogonal)
        e_interf = Proj_to_S_subspace - s_target

        #by orthogonality principle we have that e=x-x_hat is orthogonal to the subspace that we projected our data(estimates==x)
        e_artif = target_estimate - Proj_to_S_subspace

    else:
        Rss_inv = np.linalg.inv( np.dot(S.T,S) ) 
        tmp = np.array( list( map( lambda x : np.dot(target_estimate,x)  , S.T ) ) )
        c = np.dot( Rss_inv,tmp )

        Proj_to_S_subspace = np.dot(S,c)

        e_interf = Proj_to_S_subspace - s_target
        e_artif = target_estimate - Proj_to_S_subspace


    if win<len(target_estimate):
        #Calclating metrics FRAMEWISE

        all_signal_components = [ s_target , e_interf , e_artif ] 
        #padding the components in order for their lenth to be divisible with the signal length:       
        all_signal_components_PADDED = list( map( lambda component : padding_or_not(component,win,padding_flag) , all_signal_components ) ) 

        #creating the inds of the segments:
        L_new = len(all_signal_components_PADDED[0])
        all_inds = np.arange(L_new)

        b = int(L_new/win)
        #The rows of S contains the inds for the segments
        S = np.reshape(all_inds, (b,win),order="C")

        metrics = list(  map( lambda inds : compute_energy_ratios( all_signal_components_PADDED[0][inds],all_signal_components_PADDED[1][inds],all_signal_components_PADDED[2][inds] ) , S ) ) 
        
    else:
        #computing the energy ratios FRAME-WISE

        #computing the energy ratios in the WHOLE SIGNAL
        metrics = compute_energy_ratios(s_target,e_interf,e_artif)

    return metrics



#-------------------------------------------------------------------------------------------------------------------------------

def evaluate_using_museval(musdb_track,target,mono_estimate,dur,win=None,hop=None,aggregation="median"):
    #in order to compare with the custom metrics..

    #using museval.evaluate method (not eval_msdb_track)
    # it expects reference(nsrc,samples,chanels) and estimates(nsrc,samples,chanels) 

    
    if dur:
        mono_target = librosa.to_mono(musdb_track.targets[target].audio.T)[50*musdb_track.rate:(50+dur)*musdb_track.rate]
    else:
        mono_target = librosa.to_mono(musdb_track.targets[target].audio.T)        

    keys = ["SDR","ISR","SIR","SAR"]

    if win:

        #Calclating metrics FRAMEWISE
        tmp1 = museval.evaluate(np.reshape(librosa.to_mono(mono_target[:len(mono_estimate)]),(1,len(mono_estimate),1)) 
        , np.reshape(mono_estimate,(1,len(mono_estimate),1)), win=win, hop=hop, mode='v4', padding=True)

        #calculating median (or mean) over frames
        tmp1 = list(map(lambda x : ( x[0] , np.median(x[1]) if aggregation=="median" else np.mean(x[1]) ) , zip(keys,tmp1)  ) )


        
    else:
        #Calclating metrics over the WHOLE signal
        tmp1 = museval.evaluate(np.reshape(librosa.to_mono(mono_target[:len(mono_estimate)]),(1,len(mono_estimate),1)) 
        , np.reshape(mono_estimate,(1,len(mono_estimate),1)), win=len(mono_estimate), hop=len(mono_estimate), mode='v4', padding=True)

        #altertnative:
        #tmp1 = museval.metrics.bss_eval(np.reshape(mono_target[:len(mono_estimate)],(1,len(mono_estimate),1)) 
        # , np.reshape(mono_estimate,(1,len(mono_estimate),1)), window=np.inf )[:4]

        tmp1 = list( map( lambda x : (x[0] , x[1])  , zip(keys,tmp1) ) )





    return tmp1