This is basically  for testing and validating changes to docling codebase, which can be tested by downloading the folders in the "site-packages" folder and either create a new environement and 
place these folders in the same folder where your packages installed via pip get installed to or directly replace the respective docling folders within your existing environments site-packages folder.

After that you may need to install 1 or two dependencies, "accelerate" and "flash-attn2" are the main ones. Within the data folder is a folder with academic journal articles , which are to be used to test the 
stability of the docling pipeline when testing the scripts in the "scripts" folder, which include:

- scripts/docling_extract_formulas_debug.py
- scripts/docling_extract_formulas_debug_mp_mult.py
- scripts/docling_extract_formulas_mp_multi.py
- scripts/docling_test_single_GPU.py
- scripts/docling_test_nb.ipynb

  As you can deduce from the file names, two of these are meant for testing docling across multiple GPUs. Ive only tested it on 2 max, for some reason it causes a CUDA error when i run it on a chinese 4090d.
  This may be an issue specific to that 4090d GPU however, i tested this on a single 12gb 4070 super it ran just fine.

  For debugging if you want docling to save the formula snippets and the images showing the bounding boxes on the layouts its something you need to uncomment this part:
  https://github.com/agodinezmm2007/docling_mod/blob/ea18bf4a42373318ed9d108c4ca8d597a19a1151/site-packages/docling/datamodel/base_models.py#L341

  but that is to see the masked pages from where the formula snippets are generated. to save and export the actual formula snippets themselves you uncomment a line somewhere around here:

  https://github.com/agodinezmm2007/docling_mod/blob/ea18bf4a42373318ed9d108c4ca8d597a19a1151/site-packages/docling_ibm_models/code_formula_model/code_formula_predictor.py#L280

  Also, the data folder includes two log files to validate the data processing.

  If someone wanted to, they could just clone the repository, get the packages where they need to go, install missing dependencies, then run the test through the jupyer notebook you already get 190 journal articles converted to markdown, along
  with associated metadata. What you chose to do with that is up to you.

  There are a few things to look out for, it seems that when docling tries to process pdf articles having to do with prompting it will completely stop the system. It still treats some pages as tables i havent had a chance to go back and troubleshoot
  that. Formula extraction uses SmolDocling so it should be able to be used in smaller GPUs, running it on a 12gb 4070 Super and processing 50 articles confirms this. However, more VRAM will always equal more better. 
