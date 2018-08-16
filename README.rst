Install requirement
--------------------------------------------
In order to use opencv-python,gdspy package,
add following in .bashrc/.zshrc etc:

export PYTHONPATH=$PYTHONPATH:/gpfs/users/yinhuang/.local/lib/python2.7/site-packages/cv2
export PYTHONPATH=$PYTHONPATH:/gpfs/users/yinhuang/.local/lib/python2.7/site-packages/gdspy-1.2.1-py2.7-linux-x86_64.egg

Usage
--------------------------------------------
* Manual:

    1. Get edges with vertice number in foo.png
       python image2gds.py -i target.png            
       
    2. Manual input polygon seq based on foo.png and output gds
       python image2gds.py -i target.png -s seq -o out.gds      
    
* Auto(Only for HV mode with clear separation edge):

       python image2gds.py -i target.png  -o out.gds      

* image2gds_v2.py
  use opencv to do contour extraction 
