#!/bin/bash

set -e

#Number of folds
k_fold=4

#The detector you want to use
detector=res18

#How many epochs you want to train the model
epochs=200

#How frequent do you want to save the epoch
save_freq=5

#What is the batch size
batch_size=16

#Initial learning rate, decay 0.1 every 50% and 80% of epochs
lr=0.01

#Where to start the fold, used for debugging or error occurred
start_fold=0

#Store the ubmission.csv file
submission="submission"

#Split data into folds
python split.py --k-fold $k_fold

for((i=${start_fold};i<${k_fold};i++))
do
	#Train the detector
	cmd="python main.py --optim adam --model $detector --save-dir $i --train-filename ./split/train_${i}.npy --val-filename ./split/val_${i}.npy --lr $lr --workers 16 -b $batch_size --epochs $epochs --save-freq 5"
	echo "$cmd"
	echo "Training $i folder ..."
	$cmd
	errono=$?
	if [ $errono -ne 0 ]
	then
		echo "Error occurred when training ${i}th fold"
		exit $errono
	fi
	
	#Test the detector on validation
	for((j=100;j<=${epochs};j+=5))
	do
		cmd="python main.py --model $detector --resume results/${i}/${j}.ckpt --workers 1 --test-filename ./split/val_${i}.npy --test 1 --save-dir val/${i}/${j}"
		echo "$cmd"
		echo "Predicting $i folder ..."
		$cmd
		errono=$?
		if [ $errono -ne 0 ]
		then
			echo "Error occurred when testing ${i}th fold"
			exit $errono
		fi
	done

	#Find the best epoch
	cmd="python choose_best_epoch.py ${i}"
	echo $cmd
	res=`$cmd`
        if [ $errono -ne 0 ]
        then
                echo "Error occurred when finding best epoch of ${i}th fold"
		exit $errono
        fi
	best_epoch=${res: -3}
	#best_epoch=$epochs
	echo "Best epoch: $best_epoch"

	#Test on the test dataset
	cmd="python main.py --model $detector --resume results/${i}/${best_epoch}.ckpt --workers 1 --test-filename ./split/test_${i}.npy --test 1 --save-dir test/${i} --n_test 4"
	echo "$cmd"
        $cmd
	errono=$?
	if [ $errono -ne 0 ]
        then
                echo "Error occurred when finding best epoch of ${i}th fold"
                exit $errono
        fi
	
	
	if [ ! -d "$submission" ]
	then
		mkdir "$submission"
	fi
	#Generate the submission file for the test subset
	cmd="python generate_submission.py --save-dir ${submission}/${i}.csv --filename ./split/test_${i}.npy --pred-dir results/test/${i}/bbox/test_$best_epoch/"
	echo "$cmd"
	$cmd
	errono=$?
	if [ $errono -ne 0 ]
	then
		echo "Error occurred when generating submission for ${i}th fold"
		exit $errono
	fi

	#Add the submission file of the test subset to the general submission.csv file
	cat ${submission}/${i}.csv >> "${submission}.csv"
done

