
# DOSS(cnt) algorithm
mkdir -p "./results/SDSGDYCK/EASY36"
mkdir -p "./results/SDSGDYCK/NEW36"
bash pbseasy.sh "-s SDSGDYCK -d 36 -e 1850 -t 30 -r ./results/SDSGDYCK/EASY36" SDSGDYCK

mkdir -p "./results/SDSGDYCK/EASY48"
mkdir -p "./results/SDSGDYCK/NEW48"
bash pbseasy.sh "-s SDSGDYCK -d 48 -e 2450 -t 30 -r ./results/SDSGDYCK/EASY48" SDSGDYCK

mkdir -p "./results/SDSGDYCK/EASY60"
mkdir -p "./results/SDSGDYCK/NEW60"
bash pbseasy.sh "-s SDSGDYCK -d 60 -e 3050 -t 30 -r ./results/SDSGDYCK/EASY60" SDSGDYCK

mkdir -p "./results/SDSGDYCK/Rover3000"
python3 test.py -p RoverTrajPlan -d 60 -e 3000 -s SDSGDYCK -t 30 -r ./results/SDSGDYCK/Rover3000

mkdir -p "./results/SDSGDYCK/RobotPushing"
python3 test.py -p RobotPushing -d 14 -e 3000 -s SDSGDYCK -t 30 -r ./results/SDSGDYCK/RobotPushing


# DOSS(std) algorithm
mkdir -p "./results/SDSGDYCK_std/EASY36"
mkdir -p "./results/SDSGDYCK_std/NEW36"
bash pbseasy.sh "-s SDSGDYCK_std -d 36 -e 1850 -t 30 -r ./results/SDSGDYCK_std/EASY36" SDSGDYCK_std

mkdir -p "./results/SDSGDYCK_std/EASY48"
mkdir -p "./results/SDSGDYCK_std/NEW48"
bash pbseasy.sh "-s SDSGDYCK_std -d 48 -e 2450 -t 30 -r ./results/SDSGDYCK_std/EASY48" SDSGDYCK_std

mkdir -p "./results/SDSGDYCK_std/EASY60"
mkdir -p "./results/SDSGDYCK_std/NEW60"
bash pbseasy.sh "-s SDSGDYCK_std -d 60 -e 3050 -t 30 -r ./results/SDSGDYCK_std/EASY60" SDSGDYCK_std

mkdir -p "./results/SDSGDYCK_std/Rover3000"
python3 test.py -p RoverTrajPlan -d 60 -e 3000 -s SDSGDYCK_std -t 30 -r ./results/SDSGDYCK_std/Rover3000

mkdir -p "./results/SDSGDYCK_std/RobotPushing"
python3 test.py -p RobotPushing -d 14 -e 3000 -s SDSGDYCK_std -t 30 -r ./results/SDSGDYCK_std/RobotPushing


#DYCORS algorithm
mkdir -p "./results/DYCORS/EASY36"
mkdir -p "./results/DYCORS/NEW36"
bash pbseasy.sh "-s DYCORS -d 36 -e 1850 -t 30 -r ./results/DYCORS/EASY36" DYCORS

mkdir -p "./results/DYCORS/EASY48"
mkdir -p "./results/DYCORS/NEW48"
bash pbseasy.sh "-s DYCORS -d 48 -e 2450 -t 30 -r ./results/DYCORS/EASY48" DYCORS

mkdir -p "./results/DYCORS/EASY60"
mkdir -p "./results/DYCORS/NEW60"
bash pbseasy.sh "-s DYCORS -d 60 -e 3050 -t 30 -r ./results/DYCORS/EASY60" DYCORS

mkdir -p "./results/DYCORS/Rover3000"
python3 test.py -p RoverTrajPlan -d 60 -e 3000 -s DYCORS -t 30 -r ./results/DYCORS/Rover3000

mkdir -p "./results/DYCORS/RobotPushing"
python3 test.py -p RobotPushing -d 14 -e 3000 -s DYCORS -t 30 -r ./results/DYCORS/RobotPushing


#RBFOpt algorithm
mkdir -p "./results/RBFOpt/EASY36"
mkdir -p "./results/RBFOpt/NEW36"
bash pbseasy.sh "-s RBFOpt -d 36 -e 1850 -t 30 -r ./results/RBFOpt/EASY36" RBFOpt

mkdir -p "./results/RBFOpt/EASY48"
mkdir -p "./results/RBFOpt/NEW48"
bash pbseasy.sh "-s RBFOpt -d 48 -e 2450 -t 30 -r ./results/RBFOpt/EASY48" RBFOpt

mkdir -p "./results/RBFOpt/EASY60"
mkdir -p "./results/RBFOpt/NEW60"
bash pbseasy.sh "-s RBFOpt -d 60 -e 3050 -t 30 -r ./results/RBFOpt/EASY60" RBFOpt

mkdir -p "./results/RBFOpt/Rover3000"
python3 test.py -p RoverTrajPlan -d 60 -e 3000 -s RBFOpt -t 30 -r ./results/RBFOpt/Rover3000

mkdir -p "./results/RBFOpt/RobotPushing"
python3 test.py -p RobotPushing -d 14 -e 3000 -s RBFOpt -t 30 -r ./results/RBFOpt/RobotPushing

#TuRBO algorithm
mkdir -p "./results/TuRBO/EASY36"
mkdir -p "./results/TuRBO/NEW36"
bash pbseasy.sh "-s TuRBO -d 36 -e 1850 -t 30 --num_tr 1 -r ./results/TuRBO/EASY36" TuRBO

mkdir -p "./results/TuRBO/EASY48"
mkdir -p "./results/TuRBO/NEW48"
bash pbseasy.sh "-s TuRBO -d 48 -e 2450 -t 30 --num_tr 1 -r ./results/TuRBO/EASY48" TuRBO

mkdir -p "./results/TuRBO/EASY60"
mkdir -p "./results/TuRBO/NEW60"
bash pbseasy.sh "-s TuRBO -d 60 -e 3050 -t 30 --num_tr 1 -r ./results/TuRBO/EASY60" TuRBO

mkdir -p "./results/TuRBO/Rover3000"
python3 test.py -p RoverTrajPlan -d 60 -e 3000 -s TuRBO -t 30 --num_tr 1 -r ./results/TuRBO/Rover3000

mkdir -p "./results/TuRBO/RobotPushing"
python3 test.py -p RobotPushing -d 14 -e 3000 -s TuRBO -t 30 --num_tr 1 -r ./results/RBFOpt/RobotPushing


#DOSS_SLHD algorithm
mkdir -p "./results/SDSGDYCK_SLHD/EASY36"
mkdir -p "./results/SDSGDYCK_SLHD/NEW36"
bash pbseasy.sh "-s SDSGDYCK_SLHD -d 36 -e 1850 -t 30 -r ./results/SDSGDYCK_SLHD/EASY36" SDSGDYCK_SLHD

mkdir -p "./results/SDSGDYCK_SLHD/EASY48"
mkdir -p "./results/SDSGDYCK_SLHD/NEW48"
bash pbseasy.sh "-s SDSGDYCK_SLHD -d 48 -e 2450 -t 30 -r ./results/SDSGDYCK_SLHD/EASY48" SDSGDYCK_SLHD

mkdir -p "./results/SDSGDYCK_SLHD/EASY60"
mkdir -p "./results/SDSGDYCK_SLHD/NEW60"
bash pbseasy.sh "-s SDSGDYCK_SLHD -d 60 -e 3050 -t 30 -r ./results/SDSGDYCK_SLHD/EASY60" SDSGDYCK_SLHD

#SDSG algorithm
mkdir -p "./results/SDSG/EASY36"
mkdir -p "./results/SDSG/NEW36"
bash pbseasy.sh "-s SDSG -d 36 -e 1850 -t 30 -r ./results/SDSG/EASY36" SDSG

mkdir -p "./results/SDSG/EASY48"
mkdir -p "./results/SDSG/NEW48"
bash pbseasy.sh "-s SDSG -d 48 -e 2450 -t 30 -r ./results/SDSG/EASY48" SDSG

mkdir -p "./results/SDSG/EASY60"
mkdir -p "./results/SDSG/NEW60"
bash pbseasy.sh "-s SDSG -d 60 -e 3050 -t 30 -r ./results/SDSG/EASY60" SDSG


#CK algorithm
mkdir -p "./results/CK/EASY36"
mkdir -p "./results/CK/NEW36"
bash pbseasy.sh "-s CK -d 36 -e 1850 -t 30 -r ./results/CK/EASY36" CK

mkdir -p "./results/CK/EASY48"
mkdir -p "./results/CK/NEW48"
bash pbseasy.sh "-s CK -d 48 -e 2450 -t 30 -r ./results/CK/EASY48" CK

mkdir -p "./results/CK/EASY60"
mkdir -p "./results/CK/NEW60"
bash pbseasy.sh "-s CK -d 60 -e 3050 -t 30 -r ./results/CK/EASY60" CK