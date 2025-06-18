
##########################
#### You need to change the following 3 paths to your own local path
#### If you don't want need cgra hardware to execute simulation, CHIPYARD_SOURCE_ENV can be omit.
######################### 
export CGRVOPT_PROJECT_PATH="yourpath/adora-mlir"
export CGRA_ADG_PATH=$CGRVOPT_PROJECT_PATH"/experiments/resource/cgra_adg.json"
export CHIPYARD_DIR="yourpath/chipyard"

##########################
#### ADORA projects should be placed at chipyard/generators
######################### 
export ADORA_DIR=$CHIPYARD_DIR"/generators/fdra"

##########################
#### Don't have to change paths beneath
######################### 
export CGRA_OP_FILE_PATH=$CGRA_ADG_PATH
export GeneralOpNameFile="$CGRVOPT_PROJECT_PATH/lib/DFG/Documents/GeneralOpName.txt"

export PATH=/home/jhlou/CGRVOPT/cgra-opt/build/bin:$PATH

####### CONDA IN CHIPYARD
export CHIPYARD_SOURCE_ENV="$CHIPYARD_DIR/env.sh"
source $CHIPYARD_SOURCE_ENV
conda activate $CHIPYARD_DIR/.conda-env