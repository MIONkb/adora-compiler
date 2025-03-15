
##########################
#### You need to change the following 3 paths to your local path
#### If you don't want to execute simulation, CHIPYARD_SOURCE_ENV can be omit.
######################### 
export CGRVOPT_PROJECT_PATH="/home/jhlou/CGRVOPT/cgra-opt"
export CGRA_ADG_PATH="/home/jhlou/chipyard/generators/fdra/cgra-mg/src/main/resources"
export CHIPYARD_DIR="/home/jhlou/chipyard"

##########################
#### Don't have to change paths beneath
######################### 
export CGRA_OP_FILE_PATH=$CGRA_ADG_PATH
export GeneralOpNameFile="$CGRVOPT_PROJECT_PATH/lib/DFG/Documents/GeneralOpName.txt"

export CHIPYARD_SOURCE_ENV="$CHIPYARD_DIR/env.sh"

export PATH=/home/jhlou/CGRVOPT/cgra-opt/build/bin:$PATH

#### activate conda in chipyard
source $CHIPYARD_SOURCE_ENV
conda activate $CHIPYARD_DIR/.conda-env