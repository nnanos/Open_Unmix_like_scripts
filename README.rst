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
                   #. `nsgt grrrr <https://github.com/grrrr/nsgt>`_                   
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
                
                     * Wav_folder:
                            It is the PATH of the musdb18_wav dir which have to be in the following structure:
                     
                     .. image:: Folder_structure.png
                     


                     * Target_folder: 
                            It is the PATH of the dir (which is created if not exists) in which the following files will get saved: 
                                        
                                        * Spec_seg_pair_list_train.pt: a python variable (iterable) which contains the spectrogram training input-output pairs (samples).
                                        * Spec_seg_pair_list_valid.pt: The same as before but for the validation.
                     
                                        * Dataset_Params_log.json: Log which contains the parameters of the Dataset that has been created
                                        

                     * Fs: 
                            Sampling rate in which the songs to be processed are resampled  
                     
                     * seq_dur:
                            Η διάρκεια της ακολουθίας (σε sec) των παραδειγμάτων εκπαίδευσης τα οποία τροφοδοτούμε στο δίκτυο.
                     
                     * FE_params:
                            Is the parameters of the FE (front end ή input represenation) with which we feed the Neural Net.
                            The available FEs and the corresponding arguments are:
                                   
                                          * "nsgt_grr" ::
                                          
                                                 FE_params : { front_end_name : nsgt_grr , ksi_min : 32.07 , ksi_max : 7000 , B : 187 , matrix_form : 1 }
                                          
                                          
                                          * "librosa" ::
                                                 
                                                 FE_params : { front_end_name : librosa , a : 768 , M : 1024 , support : 1024 }
                                                 
                                          * "scipy" ::
                                          
                                                 FE_params : { front_end_name : scipy , a : 768 , M : 1024 , support : 1024 }
                                                 
                                          * "STFT_custom" ::
                                          
                                                 FE_params : { front_end_name : STFT_custom , a : 768 , M : 1024 , support : 1024 }
                                                 
                                          * "NSGT_CQT" :: 
                                                 
                                                 FE_params : { front_end_name : NSGT_CQT , ksi_min : 32.07 , ksi_max : 7000 , B : 24 , matrix_form : 1 }
                                  

       
       |
       |


#. TRAIN-----------------------------------------------------------------------------------------------

       After you have created the dataset you are now ready to begin an experiment with the U-Net model and with the Front-End that you have chosen. 

          COMMAND EXAMPLE: 

              * BEGIN TRAINING ::
              
                     python train.py --root /home/nnanos/OPEN_UMX_LIKE_scripts/Spectrograms_NSGT_CQT_mine_24_bass --target bass --output /home/nnanos/OPEN_UMX_LIKE_scripts/Spectrograms_NSGT_CQT_mine_24_bass/pretr_model --epochs 1000 --batch-size 32 


              * CONTINUE TRAINING ::
              
                     python train.py --model /home/nnanos/OPEN_UMX_LIKE_scripts/Spectrograms_NSGT_CQT_mine_24/pretr_model --checkpoint /home/nnanos/OPEN_UMX_LIKE_scripts/Spectrograms_NSGT_CQT_mine_24/pretr_model --root /home/nnanos/OPEN_UMX_LIKE_scripts/Spectrograms_NSGT_CQT_mine_24 --target vocals --output /home/nnanos/OPEN_UMX_LIKE_scripts/Spectrograms_NSGT_CQT_mine_24/pretr_model --epochs 300 --batch-size 32 --nb-workers 6 
         


          ARGUMENTS EXPLANATION:
          
              * root:
                     Είναι το PATH του dir το οποίο περιέχει τα αρχεία : 
                     Spec_seg_pair_list_train.pt,Spec_seg_pair_list_valid.pt,Dataset_Params_log.json
                     και το οποίο δημιουργήθηκε με το προηγούμενο script (Create_Dataset.py). 
                     Ως εκ τούτου το path αυτό θα πρέπει να είναι ίδιο με το Target_folder (παράμετρος του  Create_Dataset.py)
                     
                     
              * output: 
                     It is the PATH of the dir (which is created if not exists) in which the following files will get saved:
             
                           * model.pth: Αρχείο απαραίτητο αν θες να χρησιμοποιήσεις το μοντέλο για inference ή για evalution

                           * model.json: Αρχείο Log το οποίο περιέχει στοιχεία για την εκπαίδευση (π.χ. trainig-validation losses, execution time, Dataset parameters, arguments του train.py script )

                           * model.chkpnt: Αρχείο απαραίτητο αν θές να κάνεις συνεχίσεις την εκπαίδευση ενός ήδη εκπαιδευμένου μοντέλου από εκεί που είχε σταματήσει


              * target:
                     It is the target source that our Neural Net will be trained to separate. 
                     It can be one of the following strings:
                            * "vocals"
                            * "drums"
                            * "bass"
                            * "other"
                     


              Βασικές Υπερπαράμετροι (hyperparameters) εκπαίδευσης:

              * epochs:
                     Εποχές τις οποίες θα εκπαιδευτεί το Νευρωνικό δίκτυο


              * batch-size:
                     Το μέγεθος του batch με το οποίο τροφοδοτούμε το δίκτυο 
                       (το πλήθος των παραδειγμάτων που του δίνουμε ώστε μετά να κάνει backprop).
                       Όσο μεγαλύτερο είναι +Με τη διαδικασία εκπαίδευσης θα βρεθεί σίγουρα ένα τοπικό ελάχιστο
                                            +Γρηγορότερη επεξεργασία καθώς εκμαιταλευόμαστε περισσότερο την GPU
                                            -Περισσότερες απαιτήσεις σε μνήμη
                     
                     
               * Υπάρχουν και άλλες αλλά καλό θα ήταν να αφεθούν στις Default τιμές:)                     


       |
       |


#. EVALUATION-------------------------------------------------------------------------------------------------------------------------

       After you have created the dataset and trained the model (with the above scripts) you are now ready to evaluate the model (compute the BSS performance metrics) with one of the available evaluation methods. In the evaluation phase the songs will be resampled and processed in a block-wise manner exactly as in the training phase.

          COMMAND EXAMPLE: ::

              python evaluate.py --method-name  CQT_mine_24_bass  --Model_dir /home/nnanos/OPEN_UMX_LIKE_scripts/Spectrograms_NSGT_CQT_mine_24_bass/pretr_model  --root_TEST_dir /home/nnanos/musdb18_wav/test  --target bass  --evaldir  /home/nnanos/OPEN_UMX_LIKE_scripts/Spectrograms_NSGT_CQT_mine_24_bass/evaldir_orig_BSS_eval  --cores 1       -eval-params  "{  aggregation_method : median , eval_mthd : BSS_evaluation , nb_chan : 1 , hop : 14700 , win : 14700 }"  



          ARGUMENTS EXPLANATION:   
          

               * method-name: 
                     Είναι το όνομα της μεθόδου που κάνουμε evaluate π.χ. μπορεί να θές να 
                                συγκρίνεις διάφορα FEs εισόδου ή διάφορες αρχιτεκτονικές δικτύων .
                                Αυτό το όνομα θα φαίνεται στα Logs (μπορείς και να μην το θέσεις).

               * Model_dir: 
                      Είναι το path για το dir που περιέχει τα απαραίτητα αρχείο για το pretrained μοντέλο.
                              (θα πρέπει να είναι το ίδιο με το argument output του train.py script)

               * root_TEST_dir: 
                      It is the PATH of the dir containing the testing wavs and it has to have the structure mentioned above.

               * evaldir: 
                     It is the PATH of the dir (which is created if not exists) in which the following files will get saved:
              
                            * Eval_Log.json: Contains the arguments of this script 
                            * scores.pickle: Contains the performance metrics in a python pickle variable (this will be used by the script below for visualizing these metrics)


              * eval-params:
                     Is the parameters regarding the evaluation method that will be used.
                     The available evaluation methods and the corresponding arguments are:

                                   * "BSS_evaluation" ::

                                          -eval-params  "{  aggregation_method : median , eval_mthd : BSS_evaluation , nb_chan : 1 , hop : 14700 , win : 14700 }"


                                   * "mir_eval" ::

                                          -eval-params  "{  aggregation_method : median , eval_mthd : mir_eval , nb_chan : 1 , hop : 14700 , win : 14700 }"

                                   * "BSSeval_custom" ::

                                          -eval-params  "{  aggregation_method : median , eval_mthd : BSSeval_custom , nb_chan : 1 , hop : 14700 , win : 14700 }"


         

       |
       |
   
#. PLOTTING EVALUATION-----------------------------------------------------------------------------------------  

       After you have finished with the above phases now you can visualize the results (performance metrics) obtained in the evaluation phase as in the photo below.
       
       * Boxplots:
              .. image:: Boxplots.png
       
       
       * Metrics Aggregated over Frames and over Tracks:
              .. image:: Agg_frames_tracks.png
              

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
          
              * Model_dir:
                     Είναι το path για το dir που περιέχει τα απαραίτητα αρχείο για το pretrained μοντέλο.
                (θα πρέπει να είναι το ίδιο με το argument output του train.py script)

              * out_filename: 
                     Είναι ένας ήχος για τον οποίο θες να πάρεις απάντηση    
                     
         |
         |
                     
         USING THE PRETRAINED MODELS THAT I HAVE TRAINED IN MY EXPERIMENTS:

       |
       |
   

Software License
============

Free software: MIT license
============
