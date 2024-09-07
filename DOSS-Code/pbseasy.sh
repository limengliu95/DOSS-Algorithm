     # PROB_SET=("Ackley" "Griewank" "Keane" "Levy" "Michalewicz" "Rastrigin" "Rosenbrock" "Schwefel" "Weierstrass" "Zakharov")
# PROB_SET=("Ackley" "Keane" "Levy" "Michalewicz" "Rastrigin" "Schwefel" "Weierstrass")
PROB_SET=("Ackley" "Eggholder" "Keane" "Levy" "Michalewicz" "Rastrigin" "Schwefel" "Weierstrass" "Branin" "Hartman3" "Hartman6" "Rana" "Schubert" "StyblinskiTang")
# PROB_SET=("Ackley" "Eggholder" "Keane" "Levy" "Rastrigin" "Schwefel" "Weierstrass")
# PROB_SET=("Ackley" "Rastrigin")
# PROB_SET=("Eggholder")
# PROB_SET=("Weierstrass")
# PROB_SET=("Keane" "Levy" "Michalewicz" "Rastrigin" "Rosenbrock" "Schwefel" "Weierstrass" "Zakharov")
# PROB_SET=("Michalewicz" "Rosenbrock" "Zakharov")
# PROB_SET=("Ackley")
# PROB_SET=("F15" "F16" "F17" "F18" "F19" "F20" "F21" "F22" "F23" "F24")
# PROB_SET=("BBOB_F15" "BBOB_F16" "BBOB_F17" "BBOB_F18" "BBOB_F19" "BBOB_F20" "BBOB_F21" "BBOB_F22" "BBOB_F23" "BBOB_F24")
# PROB_SET=("BBOB_F16" "BBOB_F17" "BBOB_F18" "BBOB_F19" "BBOB_F20" "BBOB_F21" "BBOB_F22" "BBOB_F23" "BBOB_F24")
# PROB_SET=("BBOB_F16" "BBOB_F19" "BBOB_F22" "BBOB_F23")
# PROB_SET=("Cosmo -d 9 -e 300")
# STGY_SET=("DYCORS")
# STGY_SET=("CK")
# STGY_SET=("DS")

PARA_SETTING=$1
JOB_NAME=$2
# for STGY in ${STGY_SET[@]};
# do
#     PARA_SETTING="${PARA_SETTING} -s ${STGY}"
# done

for PROB in "${PROB_SET[@]}";
do
    PARA="-p ${PROB} ${PARA_SETTING}"
    echo $PARA
    python3 -u test.py ${PARA}
    sleep 2
done
