
Assess a model pdb `model_path` or all model pdbs in `model_dir` against a native pdb `native_path`.
The aligned model(s) will be saved to `output_model_path` or in `output_model_dir`.
All the score will be saved to `score_path` if provided.

## Usage
To fix numbering and run dockq and usalign
```
python --model-root /path/to/preds --output-model-root /path/to/results \
    --native-root /path/to/gt 
```

To run iface
```
python run_structure_assess.py --model-root /path/to/preds --native-root /path/to/gt \
    --output-model-root /path/to/iface_result --run-iface
```

To generate rank-result file for predicted models
```
python run_structure_assess.py --model-root /path/to/preds --score-path /path/to/results \
    --rank-result-path /path/to/result.json  --rank-model
```
