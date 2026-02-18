#!/bin/bash

BASE_URL="https://learn.zone01oujda.ma/git"

declare -A repos=(
  ["model-selection"]="$BASE_URL/saljaoui/model-selection.git"
  ["nlp-spacy"]="$BASE_URL/saljaoui/nlp-spacy.git"
  ["nlp"]="$BASE_URL/saljaoui/nlp.git"
  ["keras-2"]="$BASE_URL/saljaoui/keras-2.git"
  ["keras"]="$BASE_URL/saljaoui/keras.git"
  ["neural-networks"]="$BASE_URL/saljaoui/neural-networks.git"
  ["training"]="$BASE_URL/saljaoui/training.git"
  ["forest-prediction"]="https://learn.zone01oujda.ma/git/ykharkha/forest-prediction.git"
  ["pipeline"]="$BASE_URL/saljaoui/pipeline.git"
  ["classification"]="$BASE_URL/saljaoui/classification.git"
  ["linear-regression"]="$BASE_URL/saljaoui/linear-regression.git"
  ["time-series"]="$BASE_URL/saljaoui/time-series.git"
  ["data-wrangling"]="$BASE_URL/saljaoui/data-wrangling.git"
)

for name in "${!repos[@]}"; do
  url="${repos[$name]}"
  echo "👉 Adding $name"

  git remote add "$name" "$url"
  git fetch "$name"

  # Merge and automatically keep incoming files on conflict
  git merge --allow-unrelated-histories "$name/main" -m "merge $name" -X theirs

  git remote remove "$name"
  echo "✅ Done: $name"
done

git push origin main