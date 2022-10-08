=======================================================================
Experiments in Music Source Separation
=======================================================================

Description
============
This repository contains some scripts that I wrote within my undergraduate thesis in order to do some experiments in the problem of
music source separation. I had inspired by the scripts implemented in the `Open unmix source separation library <https://github.com/sigsep/open-unmix-pytorch.git>`_ but I wanted to test my own Front Ends and with fewer processing time . 


* What differantiates the scripts implemented in this repository from the ones in the  `Open unmix library <https://github.com/sigsep/open-unmix-pytorch.git>`_ :

        * The spectrograms are precomputed and are of fixed duration (controlled by the param seq-dur) and not computed at training time . This change was           made in order to add the capabillity of experimenting with different front ends other than Pytorch's. For example it is possible to use the front           ends provided by  `this <https://github.com/nnanos/Time_Frequency_Analysis.git>`_ Time frequency analysis-synthesis toolbox which is also                   implemented within this thesis.
          In fact the possibillities for the different Front Ends are:
                   #. `scipy.signal.stft <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html>`_
                   #. `librosa.stft <https://librosa.org/doc/main/generated/librosa.stft.html/>`_
                   #. `nsgt cqt <https://github.com/grrrr/nsgt>`_                   
                   #. `Time-Frequency Analysis-Synthesis Toolbox <https://github.com/nnanos/Time_Frequency_Analysis>`_  (implemented within this thesis)                                    
        

        * It is possible to change the sampling rate of the songs of the dataset for faster processing.
        

        * Training is done with a U-Net model as described in 
          `this <https://www.semanticscholar.org/paper/Singing-Voice-Separation-with-Deep-U-Net-Networks-Jansson-Humphrey                                             /83ea11b45cba0fc7ee5d60f608edae9c1443861d>`_ . paper. But there is still freedom to change the model by changing appropriately the model.py                 module.
          
        * Validation and Evaluation is done exactly as in training that is in a block-processing manner.
        
        * I don't use the `musdb <https://github.com/sigsep/sigsep-mus-db>`_ parser so there are no source-augmentations.
        
        * The separation is done in the single channel therefore for the computation of the evaluation metrics we use other evaluation methods than the               basic one (`museval.eval_mus_track <https://sigsep.github.io/sigsep-mus-eval/>`_) which is used only for stereo estimates.
          In fact the possibillities for the different evaluation methods are:
                   #. `mir_eval.separation.bss_eval_sources <https://craffel.github.io/mir_eval/>`_
                   #. `museval.evaluate <https://sigsep.github.io/sigsep-mus-eval/>`_
                   #. BSS_eval_custom   (also implemented within this thesis)
        

    








============

Usage
=============


#. PREPARE THE DATA-----------------------------------------------------------------------------------------

              With this script you can create the samples that will be fed in to the Neural Network. You just have to create the musdb wav folder. With this script you can control the sampling rate of the songs to be processed and the desired Front-End that they will be transformed to.  

                 COMMAND EXAMPLE: ::

                     python Data.py -dataset-params "{ Wav_folder : /home/nnanos/musdb18_wav , Target_folder : /home/nnanos/OPEN_UMX_LIKE_scripts/Spectrograms_NSGT_CQT_mine_24_bass , target_source : bass , Fs : 14700 , seq_dur : 5 , FE_params : { front_end_name : NSGT_CQT , ksi_min : 32.07 , ksi_max : 7000 , B : 24 , matrix_form : 1 } , preproc : None }" 

                ARGUMENTS EXPLANATION:  
       
       |
       |


#. TRAIN-----------------------------------------------------------------------------------------------

       After you have created the dataset you are now ready to begin an experiment with the U-Net model and with the Front-End that you have chosen. 

          COMMAND EXAMPLE: ::

              python train.py --root /home/nnanos/OPEN_UMX_LIKE_scripts/Spectrograms_NSGT_CQT_mine_24_bass --target bass --output /home/nnanos/OPEN_UMX_LIKE_scripts/Spectrograms_NSGT_CQT_mine_24_bass/pretr_model --epochs 1000 --batch-size 32 --target bass



          ARGUMENTS EXPLANATION:


       |
       |


#. EVALUATION-------------------------------------------------------------------------------------------------------------------------

       After you have created the dataset and trained the model (with the above scripts) you are now ready to evaluate the model (compute the BSS performance metrics) with one of the available evaluation methods. In the evaluation phase the songs will be resampled and processed in a block-wise manner exactly as in the training phase.

          COMMAND EXAMPLE: ::

              python evaluate.py --method-name  CQT_mine_24_bass  --Model_dir /home/nnanos/OPEN_UMX_LIKE_scripts/Spectrograms_NSGT_CQT_mine_24_bass/pretr_model  --root_TEST_dir /home/nnanos/musdb18_wav/test  --target bass  --evaldir  /home/nnanos/OPEN_UMX_LIKE_scripts/Spectrograms_NSGT_CQT_mine_24_bass/evaldir_orig_BSS_eval  --cores 1       -eval-params  "{  aggregation_method : median , eval_mthd : BSS_evaluation , nb_chan : 1 , hop : 14700 , win : 14700 }"  



          ARGUMENTS EXPLANATION:   

       |
       |
   
#. PLOTTING EVALUATION-----------------------------------------------------------------------------------------  

       After you have finished with the above phases now you can visualize the results (performance metrics) obtained in the evaluation phase. 

          COMMAND EXAMPLE: ::
       
              python Plotting_Eval_metrics.py --evaldirs /home/nnanos/Desktop/Spectrograms_STFT_scipy/evaldir_orig_BSS_eval 


          ARGUMENTS EXPLANATION:   

       |
       |


#. INFERENCE-----------------------------------------------------------------------------------------  

       After you have finished with the training of your model you can directly use your model to perform a separation to an arbitrary wav file which either       is on your PC (local) or provide a url from youtube and perform separation on a youtube track of your preference. The input wav will be resampled at the sampling rate that the model where trained and the processing will be done in a block-wise fashion where the blocks will be of duration seq-dur (the seq-dur that was used to train the model). 

          COMMAND EXAMPLE: ::

              python perform_sep.py --Model_dir /home/nnanos/OPEN_UMX_LIKE_scripts/Spectrograms_NSGT_CQT_mine_24_bass/pretr_model --out_filename /home/nnanos/Desktop/tst.wav




          ARGUMENTS EXPLANATION:   

       |
       |
   

Software License
============

Free software: MIT license
============
