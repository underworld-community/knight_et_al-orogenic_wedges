![Image of selected models](https://github.com/underworld-community/knight_et_al-orogenic_wedges/blob/master/Fig-9.jpg)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/underworld-community/template-project/master)

About
-----
Notebook for convergence models 


Files
-----
**_Please give a quick overview of purpose of the model files/directories included in this repo._**
**_Note that while light data files are fine,  heavy data should not be included in your repository._**

File | Purpose
--- | ---
`BK-Conv_One_Layer-Erosion-SimpVersion.ipynb` | Notebook for convergence models. 
`UW2DsurfaceProcesses.py`| 2D surface processes for erosion and sedimentation, includes a linear hillslope diffusion function and velocity function. 
`Fig-9.pdf` | Model at final TS for selected cases.

Tests
-----
**_Please specify how your repository is tested for correctness._**
**_Tests are not required for `laboratory` tagged repositories, although still encouraged._**
**_All other repositories must include a test._**


Parallel Safe
-------------
**_Please specify if your model will operate in parallel, and any caveats._**

Yes, test result should be obtained in both serial and parallel operation.

Check-list
----------
- [Y] (Required) Have you replaced the above sections with your own content? 
- [Y] (Required) Have you updated the Dockerfile to point to your required UW/UWG version? 
- [ ] (Required) Have you included a working Binder badge/link so people can easily run your model?
                 You probably only need to replace `template-project` with your repo name. 
- [ ] (Optional) Have you included an appropriate image for your model? 
