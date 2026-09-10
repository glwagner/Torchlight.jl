# Torchlight column MLP validation: column_mlp_tiny_float32.h5

Generated 2026-09-10T15:48:59.

| passed | failed | unsupported | not_tested |
|---|---|---|---|
| 71 | 0 | 1 | 0 |

## Environment

- `Enzyme`: 0.13.202
- `Lux`: 1.31.4
- `Optimisers`: 0.4.9
- `Reactant`: 0.2.285
- `Zygote`: 0.7.13
- `blas_threads`: 6
- `dtype`: float32
- `fixture`: column_mlp_tiny_float32.h5
- `julia`: 1.12.7
- `n_parameters`: 166
- `source_framework`: torch
- `source_platform`: macOS-26.6.2-arm64-arm-64bit-Mach-O
- `source_version`: 2.14.0
- `widths`: [7, 11, 5, 3]

## Timings (seconds)

| step | seconds |
|---|---|
| forward_plain_first | 0.087 |
| forward_plain_warm | 0.000 |
| forward_reactant_compile | 1.136 |
| forward_reactant_run_sync_first | 0.019 |
| forward_reactant_run_sync_warm | 0.000 |
| grad_enzyme_first_total | 0.062 |
| grad_enzyme_warm | 0.000 |
| grad_reactant_compile | 13.346 |
| grad_reactant_first_total | 13.844 |
| grad_reactant_run_sync_first | 0.208 |
| grad_reactant_run_sync_warm | 0.000 |
| grad_zygote_first_total | 2.734 |
| grad_zygote_warm | 0.000 |
| setup_and_mapping | 0.906 |
| train_step_reactant_first_total | 25.272 |

## mapping

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| coverage | **passed** |  |  |  |  | MappingReport(6/6 source keys -> 6 leaves; unused=0, unmapped=0, duplicates=0, shape_mismatch=0, dtype_mismatch=0) |

## forward

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| output | **passed** | 2.980e-08 | 7.290e-08 | 0/39 | (1.000e-06, 1.000e-04) |  |
| intermediates | **passed** | 1.192e-07 | 1.146e-07 | 0/455 | (1.000e-06, 1.000e-04) | all 5 stages match |
| sensitivity_input | **passed** |  |  |  |  | max output change 0.0029383301734924316 |
| sensitivity_parameters | **passed** |  |  |  |  | max output change 0.030733377 |

## forward/stages

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| layer0_act | **passed** | 7.451e-08 | 5.038e-08 | 0/143 | (1.000e-06, 1.000e-04) |  |
| layer0_preact | **passed** | 1.192e-07 | 3.771e-08 | 0/143 | (1.000e-06, 1.000e-04) |  |
| layer1_act | **passed** | 1.192e-07 | 9.476e-08 | 0/65 | (1.000e-06, 1.000e-04) |  |
| layer1_preact | **passed** | 1.192e-07 | 1.043e-07 | 0/65 | (1.000e-06, 1.000e-04) |  |
| layer2_preact | **passed** | 5.960e-08 | 1.146e-07 | 0/39 | (1.000e-06, 1.000e-04) |  |

## domain/cases

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| basis | **passed** | 4.470e-08 | 1.891e-07 | 0/21 | (1.000e-06, 1.000e-04) | batch 7 |
| constant_minus2_b4 | **passed** | 2.980e-08 | 1.010e-07 | 0/12 | (1.000e-06, 1.000e-04) | batch 4 |
| large_1e3_b2 | **passed** | 2.980e-08 | 2.873e-08 | 0/6 | (1.000e-06, 1.000e-04) | batch 2 |
| ones_b1 | **passed** | 4.470e-08 | 1.483e-07 | 0/3 | (1.000e-06, 1.000e-04) | batch 1 |
| permuted_reference_batch | **passed** | 2.980e-08 | 7.391e-08 | 0/39 | (1.000e-06, 1.000e-04) | batch 13 |
| ramp_b3 | **passed** | 5.960e-08 | 1.464e-07 | 0/9 | (1.000e-06, 1.000e-04) | batch 3 |
| randn_b1 | **passed** | 2.980e-08 | 1.557e-07 | 0/3 | (1.000e-06, 1.000e-04) | batch 1 |
| randn_b2 | **passed** | 2.980e-08 | 1.009e-07 | 0/6 | (1.000e-06, 1.000e-04) | batch 2 |
| randn_b256 | **passed** | 5.960e-08 | 1.349e-07 | 0/768 | (1.000e-06, 1.000e-04) | batch 256 |
| randn_b32 | **passed** | 5.960e-08 | 1.556e-07 | 0/96 | (1.000e-06, 1.000e-04) | batch 32 |
| randn_b7 | **passed** | 2.980e-08 | 1.269e-07 | 0/21 | (1.000e-06, 1.000e-04) | batch 7 |
| randn_scale0.01_b8 | **passed** | 5.960e-08 | 2.231e-07 | 0/24 | (1.000e-06, 1.000e-04) | batch 8 |
| randn_scale100_b8 | **passed** | 2.980e-08 | 6.140e-08 | 0/24 | (1.000e-06, 1.000e-04) | batch 8 |
| randn_scale10_b8 | **passed** | 5.960e-08 | 9.109e-08 | 0/24 | (1.000e-06, 1.000e-04) | batch 8 |
| randn_scale1_b8 | **passed** | 4.470e-08 | 1.940e-07 | 0/24 | (1.000e-06, 1.000e-04) | batch 8 |
| reference_batch_first_half | **passed** | 2.980e-08 | 5.871e-08 | 0/18 | (1.000e-06, 1.000e-04) | batch 6 |
| reference_batch_second_half | **passed** | 2.980e-08 | 9.033e-08 | 0/21 | (1.000e-06, 1.000e-04) | batch 7 |
| zeros_b1 | **passed** | 1.490e-08 | 1.217e-07 | 0/3 | (1.000e-06, 1.000e-04) | batch 1 |

## domain

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| all_cases | **passed** | 5.960e-08 | 2.231e-07 | 0/1122 | (1.000e-06, 1.000e-04) | 18/18 input families match |
| column_independence | **passed** |  |  |  |  | PASS split_batch: max_abs=0.000e+00 max_rel=0.000e+00 nl2=0.000e+00 fails=0/39; PASS permuted_batch: max_abs=0.000e+00 max_rel=0.000e+00 nl2=0.000e+00 fails=0/39 |

## derivatives/zygote

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| mse_loss | **passed** | 1.192e-07 | 1.002e-07 | 0/1 | (1.000e-06, 1.000e-04) |  |
| grad_input | **passed** | 1.164e-09 | 9.070e-08 | 0/91 | (1.000e-06, 1.000e-04) |  |
| grad_params | **passed** | 1.490e-08 | 1.965e-07 | 0/166 | (1.000e-06, 1.000e-04) |  |
| vjp_input | **passed** | 2.235e-08 | 1.123e-07 | 0/91 | (1.000e-06, 1.000e-04) |  |
| vjp_params | **passed** | 4.768e-07 | 1.054e-07 | 0/166 | (1.000e-06, 1.000e-04) |  |

## derivatives/enzyme

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| mse_loss | **passed** | 0 | 0 | 0/1 | (1.000e-06, 1.000e-04) |  |
| grad_input | **passed** | 1.164e-09 | 9.070e-08 | 0/91 | (1.000e-06, 1.000e-04) |  |
| grad_params | **passed** | 1.490e-08 | 1.965e-07 | 0/166 | (1.000e-06, 1.000e-04) |  |
| vjp_input | **passed** | 2.235e-08 | 1.123e-07 | 0/91 | (1.000e-06, 1.000e-04) |  |
| vjp_params | **passed** | 4.768e-07 | 1.054e-07 | 0/166 | (1.000e-06, 1.000e-04) |  |

## derivatives/reactant

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| mse_loss | **passed** | 2.384e-07 | 2.004e-07 | 0/1 | (1.000e-06, 1.000e-04) |  |
| grad_input | **passed** | 2.328e-09 | 1.288e-07 | 0/91 | (1.000e-06, 1.000e-04) |  |
| grad_params | **passed** | 2.980e-08 | 3.536e-07 | 0/166 | (1.000e-06, 1.000e-04) |  |
| vjp_input | **passed** | 1.490e-08 | 9.655e-08 | 0/91 | (1.000e-06, 1.000e-04) |  |
| vjp_params | **passed** | 4.768e-07 | 1.205e-07 | 0/166 | (1.000e-06, 1.000e-04) |  |
| compiled_forward | **passed** | 4.470e-08 | 1.066e-07 | 0/39 | (1.000e-06, 1.000e-04) | PASS compiled_forward: max_abs=4.470e-08 max_rel=8.740e-07 nl2=1.066e-07 fails=0/39; perturbed-input change 0.0028969347; perturbed-parameter change 0.03587234 |

## derivatives

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| reactant_vs_zygote_grad_input | **passed** | 1.863e-09 | 1.253e-07 | 0/91 | (1.000e-06, 1.000e-04) |  |

## probes

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| finite_difference_input | **passed** | 1.175e-06 |  |  |  | best h=0.01 fd=0.022351741790771484 analytic=0.02235291711986065 rel=5.2580568471774524e-5 (bar 0.01) |
| finite_difference_params | **passed** | 2.277e-05 |  |  |  | best h=0.003 fd=-1.1650919914245605 analytic=-1.1651148 rel=1.9542258907100705e-5 (bar 0.01) |
| adjoint_consistency_cross | **passed** |  |  |  |  | input: <v,Ju>=0.1395525187253952 <Jᵀv,u>=0.13955247402191162 rel=3.2033448045110784e-7; params: -23.633222579956055 vs -23.63322 rel=8.070624420176077e-8 (bar 0.0001) |
| jvp_input_enzyme_forward | **passed** | 4.470e-08 | 1.915e-07 | 0/39 | (1.000e-06, 1.000e-04) |  |

## training

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| sgd_grads | **passed** | 1.490e-08 | 1.965e-07 | 0/166 | (1.000e-06, 1.000e-04) |  |
| sgd_params_after | **passed** | 1.490e-08 | 8.435e-09 | 0/166 | (1.000e-06, 1.000e-04) | lr=0.05;  |
| sgd_update_delta | **passed** | 1.490e-08 | 1.628e-06 | 0/166 | (1.000e-06, 1.000e-04) |  |
| adam_params_after | **passed** | 9.313e-10 | 7.236e-10 | 0/166 | (1.000e-06, 1.000e-04) | lr=0.001 betas=(0.9,0.999) eps=1.0e-8;  |
| adam_update_delta | **passed** | 9.313e-10 | 1.266e-07 | 0/166 | (1.000e-06, 1.000e-04) |  |
| adam_state_exp_avg | **passed** | 7.451e-09 | 3.153e-07 | 0/166 | (1.000e-06, 1.000e-04) |  |
| adam_state_exp_avg_sq | **passed** | 1.579e-09 | 1.308e-05 | 0/166 | (1.000e-06, 1.000e-04) |  |
| adam_state | **passed** |  |  |  |  | exp_avg, exp_avg_sq compared in Lux layout; source step counters all == 1: true |
| compiled_sgd_step_reactant | **passed** | 1.490e-08 | 8.494e-09 | 0/166 | (1.000e-06, 1.000e-04) |  |

## dropout

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| masked_forward | **passed** | 4.470e-08 | 1.105e-07 | 0/39 | (1.000e-06, 1.000e-04) | shared 0/1 masks, scale 1/(1-p) |
| masked_grad_input | **passed** | 1.397e-09 | 9.421e-08 | 0/91 | (1.000e-06, 1.000e-04) |  |
| masked_grad_params | **passed** | 2.980e-08 | 1.198e-07 | 0/166 | (1.000e-06, 1.000e-04) |  |
| native_dropout_statistics | **passed** |  |  |  |  | keep fraction 0.9010467529296875 (expected 0.9 ± 0.01), scaling 1/(1-p): true |
| native_dropout_testmode_identity | **passed** |  |  |  |  | test mode is identity |
| native_stochastic_trajectory_match | **unsupported** |  |  |  |  | framework RNG streams differ by design; compare stochastic training statistically (not run here) |

## defect_detection

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| permuted_features | **passed** |  |  |  |  | injected feature permutation flagged: true; FAIL permuted_features: max_abs=1.319e-01 max_rel=2.199e+00 nl2=3.224e-01 fails=39/39 |
| missing_bias | **passed** |  |  |  |  | zeroed first bias flagged: true; FAIL missing_bias: max_abs=1.197e-01 max_rel=5.847e+00 nl2=4.155e-01 fails=39/39 |
| wrong_loss_denominator | **passed** |  |  |  |  | wrong reduction flagged: true; FAIL wrong_loss_denominator: max_abs=2.380e+00 max_rel=2.000e+00 nl2=2.000e+00 fails=1/1 |
| detached_input_gradient | **passed** |  |  |  |  | zero input gradient flagged: true; FAIL detached_input_gradient: max_abs=1.655e-02 max_rel=1.000e+00 nl2=1.000e+00 fails=90/91 (relative_to_max) |
| incomplete_mapping_raises | **passed** |  |  |  |  | dropping one mapping entry raises: true |

## acceptance

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| required_capabilities | **passed** |  |  |  |  | all 28 required capabilities passed and no failures |

