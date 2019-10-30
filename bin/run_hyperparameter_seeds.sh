#!/bin/bash

# ./bin/run_hyperparameter_seeds.sh training_config output_prefix
#


training_config=$1
shift;
output_prefix=$1
shift;
key=$1

random_seeds=(1989894904 2294922467 2002866410 1004506748 4076792239)
numpy_seeds=(1053248695 2739105195 1071118652 755056791 3842727116)
pytorch_seeds=(81406405 807621944 3166916287 3467634827 1189731539)

i=0
while [ $i -lt 5 ]; do
    rs=${random_seeds[$i]}
    ns=${numpy_seeds[$i]}
    ps=${pytorch_seeds[$i]}

    for LR in 2e-5 3e-5 5e-5
    do
        for NUM_EPOCHS in 3 4
        do

            echo "$i $LR $NUM_EPOCHS"

            overrides="{\"random_seed\": $rs, \"numpy_seed\": $ns, \"pytorch_seed\": $ps, \"trainer\": {\"num_epochs\": $NUM_EPOCHS, \"learning_rate_scheduler\": {\"num_epochs\": $NUM_EPOCHS}, \"optimizer\": {\"lr\": $LR}}}"

            outdir=${output_prefix}_lr${LR}_${NUM_EPOCHS}epochs_SEED_$i
            allennlp train --file-friendly-logging --include-package kb.include_all $training_config -s $outdir --overrides "$overrides"
    
        done

    done

    let i=i+1
done


