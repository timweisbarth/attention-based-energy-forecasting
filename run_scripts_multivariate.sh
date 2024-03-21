#cd $WORK/thesis2
#conda activate $WORK/conda_envs/thesis2
exp_name = 'Exp1'

# multivariate
sbatch ./scripts/$exp_name/multivariate/lstm.sh
sbatch ./scripts/$exp_name/multivariate/transformer.sh
sbatch ./scripts/$exp_name/multivariate/informer.sh
sbatch ./scripts/$exp_name/multivariate/autoformer.sh
sbatch ./scripts/$exp_name/multivariate/dlinear.sh
sbatch ./scripts/$exp_name/multivariate/patchtst.sh
sbatch ./scripts/$exp_name/multivariate/tsmixer.sh

# Non deepl (uni- and multivariate)
sbatch ./scripts/xgboost.sh

#conda deactivate
