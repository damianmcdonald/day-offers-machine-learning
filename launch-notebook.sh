#!/bin/bash
#
#   ____   __          _                 _                            _                              
#  / __ \ / _|        | |               | |                          | |             /\        /\    
# | |  | | |_ ___ _ __| |_ __ _ ___     | | ___  _ __ _ __   __ _  __| | __ _ ___   /  \      /  \   
# | |  | |  _/ _ \ '__| __/ _` / __|_   | |/ _ \| '__| '_ \ / _` |/ _` |/ _` / __| / /\ \    / /\ \  
# | |__| | ||  __/ |  | || (_| \__ \ |__| | (_) | |  | | | | (_| | (_| | (_| \__ \/ ____ \  / ____ \ 
#  \____/|_| \___|_|   \__\__,_|___/\____/ \___/|_|  |_| |_|\__,_|\__,_|\__,_|___/_/    \_\/_/    \_\
#                                                                                                    
#                                                                                                    

##################################### NOTES #####################################

# Once the notebook is launched, the docker container will output a URL similar to the one shown below:
# http://127.0.0.1:8888/?token=312600380cba7ef641c4a625fd56b10a7015601332d0251c
# This URL refers to the container port NOT the host port.
# Therefore, you should modify the port number (from 8888 to 10000) prior to accessing the URL on the host.

#################################################################################

# define the notebook docker image
SCIPY_NOTEBOOK=jupyter/scipy-notebook:3b1f4f5e6cc1

# pull the image (if the image is already pulled it will skip this step)
docker pull ${SCIPY_NOTEBOOK}


# run the docker container
##  the notebook will listen on port 10000 on the host machine
##  the notebook will persist any data saved in the "work" folder onm the guest to the "notebook-data" folder on the host
docker run --rm -p 10000:8888 -e JUPYTER_ENABLE_LAB=yes -v "${PWD}/notebook-data":/home/jovyan/work ${SCIPY_NOTEBOOK}
