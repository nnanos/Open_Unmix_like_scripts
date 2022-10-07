=======================================================================
Time Frequency Analysis Toolbox
=======================================================================

Description
============
Time frequency transforms under the mathematical framework of NON STATIONARY GABOR FRAMES as described in the paper  `THEORY, IMPLEMENTATION AND APPLICATIONS OF
NONSTATIONARY GABOR FRAMES <https://www.sciencedirect.com/science/article/pii/S0377042711004900/>`_. 
IMPLEMENTED:

* STFT
.. image:: /STFT.png
    
* CQT
.. image:: /CQT.png

* SCALE_FRAMES
.. image:: /SCALE_FRAMES.png

============

Software License
============

Free software: MIT license

============

Installation
============

::

    pip install Time-Frequency-Analysis

You can also install the in-development version with::

    pip install https://github.com/nnanos/Time_Frequency_Analysis/archive/main.zip

============



Usage
=============


* TEST THE TRANSFORMS! 

  #. execution time 
  #. perfect reconstruction property
  #. Redunduncy  
  #. visualization

  Steps to follow:

  #. create the repo_dir
  #. git clone https://github.com/nnanos/Time_Frequency_Analysis.git repo_dir
  #. cd to repo_dir/src/Time_Frequency_Analysis 
  #. Examples of execution:


  For NSGT_CQT ::

      python __main__.py --front_end NSGT_CQT -p "{ ksi_s : 44100 , ksi_min : 32.07 , ksi_max : 3907.07 , B : 12 , matrix_form : 1 }" --plot_spectrograms True  
     
  For NSGT_scale_frames ::

      python __main__.py --front_end NSGT_SCALE_FRAMES -p "{ onset_det : False , ksi_s : 44100 , min_scl : 128 , ovrlp_fact : 0.5 , middle_window : np.hanning , matrix_form : 0 , multiproc : 1 }" --plot_spectrograms True
     
  For STFT ::

      python __main__.py --front_end STFT -p "{ a : 1024 , M : 4096 , support : 4096 }" --plot_spectrograms True


|
|
|

* USE THEM!:

    Use case for STFT_CUSTOM::

        # from package_name.module_name import func1,func2 -----OR----- from package_name import module_name1,module_name2
        from Audio_proc_lib.audio_proc_functions import load_music
        from Time_Frequency_Analysis import STFT_custom
        import numpy as np

        x,s = load_music()

        a = 2048
        M = 4096
        support = 4096
        g = np.hanning(support) 
        L = len(x)      

        stft = STFT_custom.STFT_CUSTOM(g,a,M,support,L)
        X = stft.forward(x)
        x_rec = stft.backward(X)   
        

    
    Use case for NSGT_CQT::

        # from package_name.module_name import func1,func2 -----OR----- from package_name import module_name1,module_name2
        from Audio_proc_lib.audio_proc_functions import load_music
        from Time_Frequency_Analysis import NSGT_CQT

        x,s = load_music()

        #NSGT cqt params----------
        ksi_min = 32.7
        ksi_max = 3951.07
        B=12
        ksi_s = s
        matrix_form = False

        nsgt = NSGT_CQT.NSGT_cqt(ksi_s,ksi_min,ksi_max,B,L,matrix_form)
        X = nsgt.forward(x)
        x_rec = nsgt.backward(X)   


    Use case for NSGT_SCALE_FRAMES::

        # from package_name.module_name import func1,func2 -----OR----- from package_name import module_name1,module_name2
        from Audio_proc_lib.audio_proc_functions import load_music
        from Time_Frequency_Analysis import SCALE_FRAMES
        import numpy as np

        x,s = load_music()

        #Scale_frame params--------------------
        min_scl = 512
        multiproc = True
        nb_processes = 6
        ovrlp_fact = 0.5
        #middle_window = sg.tukey
        middle_window = np.hanning
        matrix_form = True        

        onsets = librosa.onset.onset_detect(y=x, sr=s, units="samples")
        scale_frame_obj = SCALE_FRAMES.scale_frame(ksi_s=s,min_scl=min_scl,overlap_factor=ovrlp_fact,onset_seq=onsets,middle_window=middle_window,L=len(x),matrix_form=matrix_form,multiproc=multiproc)
            
        c = scale_frame_obj.forward(x)
        x_rec = scale_frame_obj.backward(c)
        
============
    



Documentation
=============
