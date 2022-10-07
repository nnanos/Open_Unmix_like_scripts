=======================================================================
Experiments in the Music Source Separation
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

* Machine learning: Pipeline
    In order to use the scripts for your experiments with the U-Net model architecture and with a Front End of the available, of your choice, you can           by following the steps bellow:


For a STFT Front End ::

      python __main__.py --front_end STFT -p "{ a : 1024 , M : 4096 , support : 4096 }" --plot_spectrograms True



For a CQT Front End ::

      python __main__.py --front_end NSGT_CQT -p "{ ksi_s : 44100 , ksi_min : 32.07 , ksi_max : 3907.07 , B : 12 , matrix_form : 1 }" --plot_spectrograms True  
     
     


* Inference: Perform Separation
    After you have finished with the training of your model you can directly use your model to perform a separation to an arbitrary wav file which either       is on your PC (local) or provide a url from youtube as described below: 
       
       
       


Software License
============

Free software: MIT license
============
