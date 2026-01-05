# Credit Score Projec

educational ml scoring project

pytest -q


python -m src.models.train \
  --data data/processed/credit_default.csv \
  --target target \
  --model logreg \
  --n-iter 30 \
  --cv 5 \
  --experiment Credit_Default_Prediction

mlflow ui 

great_expectations datasource new
great_expectations suite new
great_expectations checkpoint new credit_default_checkpoint

black .
flake8 src tests
pytest
export GX_BASE_DIR="$(pwd)"
great_expectations checkpoint run credit_default_checkpoint


