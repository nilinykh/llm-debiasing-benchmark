### Plotting commands for a single dataset/annotation/condition

Use the same script for both conditions. The output is written under `results/vary-expert-single/<condition>/`.

```bash
python lib/vary_expert_plot_single.py ../results/amazon-bert-surprisal/vary-num-expert-plain-surprisal --dataset amazon --annotation bert --condition surprisal-text

python lib/vary_expert_plot_single.py ../hf_llm_debiasing_benchmark/experiments/vary-expert --dataset amazon --annotation bert --condition baseline
```

### Data shape note

The original HF baseline tensors have shape `(500, 10, 5)`.

- `500` = repetitions. Each repetition is one independent random resampling run of expert-labeled subset selection, driven by the seed and the random expert sample draw.
- `10` = expert-budget grid points. These are the logarithmically spaced values of `n` (number of experts) from 200 up to the dataset size.
- `5` = model parameters per fit. For logistic regression this is 4 feature coefficients plus the intercept.