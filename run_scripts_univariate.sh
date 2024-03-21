#cd $WORK/thesis2
#conda activate $WORK/conda_envs/thesis2
exp_name = 'Exp1'
# univariate
sbatch ./scripts/$exp_name/univariate/lstm.sh
sbatch ./scripts/$exp_name/univariate/transformer.sh
sbatch ./scripts/$exp_name/univariate/informer.sh
sbatch ./scripts/$exp_name/univariate/autoformer.sh
sbatch ./scripts/$exp_name/univariate/dlinear.sh
sbatch ./scripts/$exp_name/univariate/patchtst.sh
sbatch ./scripts/$exp_name/univariate/tsmixer.sh

#conda deactivate
