
#cd $WORK/thesis2
#conda activate $WORK/conda_envs/thesis2

# univariate
sbatch ./scripts/univariate/lstm.sh
sbatch ./scripts/univariate/transformer.sh
sbatch ./scripts/univariate/informer.sh
sbatch ./scripts/univariate/autoformer.sh

# multivariate
sbatch ./scripts/multivariate/lstm.sh
sbatch ./scripts/multivariate/transformer.sh
sbatch ./scripts/multivariate/informer.sh
sbatch ./scripts/multivariate/autoformer.sh

# Non deepl (uni- and multivariate)
sbatch ./scripts/xgboost.sh

#conda deactivate
