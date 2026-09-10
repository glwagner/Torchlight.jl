# Torchlight column MLP validation: column_mlp_full_float32.h5

Generated 2026-09-10T15:47:19.

| passed | failed | unsupported | not_tested |
|---|---|---|---|
| 73 | 3 | 1 | 0 |

## Environment

- `Enzyme`: 0.13.202
- `Lux`: 1.31.4
- `Optimisers`: 0.4.9
- `Reactant`: 0.2.285
- `Zygote`: 0.7.13
- `blas_threads`: 6
- `dtype`: float32
- `fixture`: column_mlp_full_float32.h5
- `julia`: 1.12.7
- `n_parameters`: 1214876
- `source_framework`: torch
- `source_platform`: macOS-26.6.2-arm64-arm-64bit-Mach-O
- `source_version`: 2.14.0
- `widths`: [420, 512, 512, 512, 512, 412]

## Timings (seconds)

| step | seconds |
|---|---|
| forward_plain_first | 0.105 |
| forward_plain_warm | 0.000 |
| forward_reactant_compile | 1.320 |
| forward_reactant_run_sync_first | 0.030 |
| forward_reactant_run_sync_warm | 0.001 |
| grad_enzyme_first_total | 0.094 |
| grad_enzyme_warm | 0.007 |
| grad_reactant_compile | 14.139 |
| grad_reactant_first_total | 14.678 |
| grad_reactant_run_sync_first | 0.243 |
| grad_reactant_run_sync_warm | 0.002 |
| grad_zygote_first_total | 4.859 |
| grad_zygote_warm | 0.002 |
| setup_and_mapping | 0.967 |
| train_step_reactant_first_total | 27.598 |

## mapping

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| coverage | **passed** |  |  |  |  | MappingReport(10/10 source keys -> 10 leaves; unused=0, unmapped=0, duplicates=0, shape_mismatch=0, dtype_mismatch=0) |

## forward

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| output | **passed** | 2.459e-07 | 8.776e-07 | 0/26368 | (1.000e-06, 1.000e-04) |  |
| intermediates | **passed** | 9.537e-07 | 8.639e-07 | 0/288512 | (1.000e-06, 1.000e-04) | all 9 stages match |
| sensitivity_input | **passed** |  |  |  |  | max output change 0.0021115094423294067 |
| sensitivity_parameters | **passed** |  |  |  |  | max output change 0.1853904 |

## forward/stages

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| layer0_act | **passed** | 5.066e-07 | 1.746e-07 | 0/32768 | (1.000e-06, 1.000e-04) |  |
| layer0_preact | **passed** | 9.537e-07 | 1.790e-07 | 0/32768 | (1.000e-06, 1.000e-04) |  |
| layer1_act | **passed** | 5.364e-07 | 3.985e-07 | 0/32768 | (1.000e-06, 1.000e-04) |  |
| layer1_preact | **passed** | 6.557e-07 | 4.105e-07 | 0/32768 | (1.000e-06, 1.000e-04) |  |
| layer2_act | **passed** | 4.768e-07 | 5.987e-07 | 0/32768 | (1.000e-06, 1.000e-04) |  |
| layer2_preact | **passed** | 5.364e-07 | 6.030e-07 | 0/32768 | (1.000e-06, 1.000e-04) |  |
| layer3_act | **passed** | 3.278e-07 | 7.749e-07 | 0/32768 | (1.000e-06, 1.000e-04) |  |
| layer3_preact | **passed** | 3.278e-07 | 7.755e-07 | 0/32768 | (1.000e-06, 1.000e-04) |  |
| layer4_preact | **passed** | 2.431e-07 | 8.639e-07 | 0/26368 | (1.000e-06, 1.000e-04) |  |

## domain/cases

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| basis | **passed** | 1.863e-07 | 7.019e-07 | 0/173040 | (1.000e-06, 1.000e-04) | batch 420 |
| constant_minus2_b4 | **passed** | 2.161e-07 | 9.515e-07 | 0/1648 | (1.000e-06, 1.000e-04) | batch 4 |
| large_1e3_b2 | **passed** | 3.874e-07 | 1.217e-06 | 0/824 | (1.000e-06, 1.000e-04) | batch 2 |
| ones_b1 | **passed** | 1.788e-07 | 8.060e-07 | 0/412 | (1.000e-06, 1.000e-04) | batch 1 |
| permuted_reference_batch | **passed** | 2.459e-07 | 8.776e-07 | 0/26368 | (1.000e-06, 1.000e-04) | batch 64 |
| ramp_b3 | **passed** | 1.863e-07 | 8.589e-07 | 0/1236 | (1.000e-06, 1.000e-04) | batch 3 |
| randn_b1 | **passed** | 1.276e-07 | 7.415e-07 | 0/412 | (1.000e-06, 1.000e-04) | batch 1 |
| randn_b2 | **passed** | 1.937e-07 | 8.328e-07 | 0/824 | (1.000e-06, 1.000e-04) | batch 2 |
| randn_b256 | **passed** | 2.906e-07 | 8.699e-07 | 0/105472 | (1.000e-06, 1.000e-04) | batch 256 |
| randn_b32 | **passed** | 2.533e-07 | 8.847e-07 | 0/13184 | (1.000e-06, 1.000e-04) | batch 32 |
| randn_b7 | **passed** | 2.831e-07 | 8.687e-07 | 0/2884 | (1.000e-06, 1.000e-04) | batch 7 |
| randn_scale0.01_b8 | **passed** | 1.192e-07 | 7.233e-07 | 0/3296 | (1.000e-06, 1.000e-04) | batch 8 |
| randn_scale100_b8 | **passed** | 3.278e-07 | 8.555e-07 | 0/3296 | (1.000e-06, 1.000e-04) | batch 8 |
| randn_scale10_b8 | **passed** | 3.129e-07 | 8.732e-07 | 0/3296 | (1.000e-06, 1.000e-04) | batch 8 |
| randn_scale1_b8 | **passed** | 2.235e-07 | 8.985e-07 | 0/3296 | (1.000e-06, 1.000e-04) | batch 8 |
| reference_batch_first_half | **passed** | 2.459e-07 | 8.741e-07 | 0/13184 | (1.000e-06, 1.000e-04) | batch 32 |
| reference_batch_second_half | **passed** | 2.235e-07 | 8.811e-07 | 0/13184 | (1.000e-06, 1.000e-04) | batch 32 |
| zeros_b1 | **passed** | 7.451e-08 | 5.621e-07 | 0/412 | (1.000e-06, 1.000e-04) | batch 1 |

## domain

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| all_cases | **passed** | 3.874e-07 | 1.217e-06 | 0/366268 | (1.000e-06, 1.000e-04) | 18/18 input families match |
| column_independence | **passed** |  |  |  |  | PASS split_batch: max_abs=0.000e+00 max_rel=0.000e+00 nl2=0.000e+00 fails=0/26368; PASS permuted_batch: max_abs=0.000e+00 max_rel=0.000e+00 nl2=0.000e+00 fails=0/26368 |

## derivatives/zygote

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| mse_loss | **passed** | 0 | 0 | 0/1 | (1.000e-06, 1.000e-04) |  |
| grad_input | **passed** | 1.728e-11 | 8.665e-07 | 0/26880 | (1.000e-06, 1.000e-04) |  |
| grad_params | **passed** | 5.821e-10 | 7.981e-07 | 0/1214876 | (1.000e-06, 1.000e-04) |  |
| vjp_input | **passed** | 1.490e-07 | 6.261e-07 | 0/26880 | (1.000e-06, 1.000e-04) |  |
| vjp_params | **passed** | 3.815e-06 | 8.044e-07 | 0/1214876 | (1.000e-06, 1.000e-04) |  |

## derivatives/enzyme

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| mse_loss | **passed** | 5.960e-08 | 5.968e-08 | 0/1 | (1.000e-06, 1.000e-04) |  |
| grad_input | **passed** | 1.728e-11 | 8.665e-07 | 0/26880 | (1.000e-06, 1.000e-04) |  |
| grad_params | **passed** | 5.821e-10 | 7.981e-07 | 0/1214876 | (1.000e-06, 1.000e-04) |  |
| vjp_input | **passed** | 1.490e-07 | 6.261e-07 | 0/26880 | (1.000e-06, 1.000e-04) |  |
| vjp_params | **passed** | 3.815e-06 | 8.044e-07 | 0/1214876 | (1.000e-06, 1.000e-04) |  |

## derivatives/reactant

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| mse_loss | **passed** | 0 | 0 | 0/1 | (1.000e-06, 1.000e-04) |  |
| grad_input | **passed** | 1.546e-11 | 8.713e-07 | 0/26880 | (1.000e-06, 1.000e-04) |  |
| grad_params | **passed** | 2.910e-10 | 8.380e-07 | 0/1214876 | (1.000e-06, 1.000e-04) |  |
| vjp_input | **passed** | 1.788e-07 | 6.614e-07 | 0/26880 | (1.000e-06, 1.000e-04) |  |
| vjp_params | **passed** | 3.457e-06 | 8.491e-07 | 0/1214876 | (1.000e-06, 1.000e-04) |  |
| compiled_forward | **passed** | 2.533e-07 | 8.880e-07 | 0/26368 | (1.000e-06, 1.000e-04) | PASS compiled_forward: max_abs=2.533e-07 max_rel=7.214e-02 nl2=8.880e-07 fails=0/26368; perturbed-input change 0.0023951456; perturbed-parameter change 0.2213872 |

## derivatives

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| reactant_vs_zygote_grad_input | **passed** | 1.774e-11 | 8.625e-07 | 0/26880 | (1.000e-06, 1.000e-04) |  |

## probes

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| finite_difference_input | **passed** | 8.848e-08 |  |  |  | best h=0.1 fd=0.0009295344352722168 analytic=0.0009296229109168053 rel=9.517369198787788e-5 (bar 0.01) |
| finite_difference_params | **passed** | 2.782e-04 |  |  |  | best h=0.001 fd=-0.03504753112792969 analytic=-0.034769285 rel=0.00800262800685159 (bar 0.01) |
| adjoint_consistency_cross | **passed** |  |  |  |  | input: <v,Ju>=-7.132602691650391 <Jᵀv,u>=-7.132615566253662 rel=1.8050325510879365e-6; params: -614.4890747070312 vs -614.4881 rel=1.589226790509813e-6 (bar 0.0001) |
| jvp_input_enzyme_forward | **passed** | 1.639e-07 | 7.749e-07 | 0/26368 | (1.000e-06, 1.000e-04) |  |

## training

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| sgd_grads | **passed** | 5.821e-10 | 7.981e-07 | 0/1214876 | (1.000e-06, 1.000e-04) |  |
| sgd_params_after | **passed** | 3.725e-09 | 4.278e-09 | 0/1214876 | (1.000e-06, 1.000e-04) | lr=0.05;  |
| sgd_update_delta | **passed** | 3.725e-09 | 2.501e-05 | 0/1214876 | (1.000e-06, 1.000e-04) |  |
| adam_params_after | **failed** | 6.866e-06 | 1.150e-06 | 1/1214876 | (1.000e-06, 1.000e-04) | lr=0.001 betas=(0.9,0.999) eps=1.0e-8;  worst block layers.3.weight: FAIL layers.3.weight: max_abs=6.866e-06 max_rel=2.813e-03 nl2=1.037e-06 fails=1/262144 (relative_to_max) |
| adam_update_delta | **failed** | 6.866e-06 | 2.939e-05 | 102/1214876 | (1.000e-06, 1.000e-04) |  worst block layers.3.weight: FAIL layers.3.weight: max_abs=6.866e-06 max_rel=1.318e-01 nl2=2.651e-05 fails=18/262144 (relative_to_max) |
| adam_update_delta_diagnosis | **passed** |  |  |  |  | 102 of 102 failing update components have \|grad\| < 1e3*eps=9.99999993922529e-6: consistent with Float32 ill-conditioning of g/(\|g\|+eps), not a rule mismatch |
| adam_state_exp_avg | **passed** | 8.731e-11 | 8.337e-07 | 0/1214876 | (1.000e-06, 1.000e-04) |  |
| adam_state_exp_avg_sq | **passed** | 7.017e-14 | 1.301e-05 | 0/1214876 | (1.000e-06, 1.000e-04) |  |
| adam_state | **passed** |  |  |  |  | exp_avg, exp_avg_sq compared in Lux layout; source step counters all == 1: true |
| compiled_sgd_step_reactant | **passed** | 3.725e-09 | 3.028e-09 | 0/1214876 | (1.000e-06, 1.000e-04) |  |

## dropout

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| masked_forward | **passed** | 3.129e-07 | 8.503e-07 | 0/26368 | (1.000e-06, 1.000e-04) | shared 0/1 masks, scale 1/(1-p) |
| masked_grad_input | **passed** | 2.183e-11 | 8.481e-07 | 0/26880 | (1.000e-06, 1.000e-04) |  |
| masked_grad_params | **passed** | 4.657e-10 | 7.684e-07 | 0/1214876 | (1.000e-06, 1.000e-04) |  |
| native_dropout_statistics | **passed** |  |  |  |  | keep fraction 0.9010467529296875 (expected 0.9 ± 0.01), scaling 1/(1-p): true |
| native_dropout_testmode_identity | **passed** |  |  |  |  | test mode is identity |
| native_stochastic_trajectory_match | **unsupported** |  |  |  |  | framework RNG streams differ by design; compare stochastic training statistically (not run here) |

## defect_detection

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| permuted_features | **passed** |  |  |  |  | injected feature permutation flagged: true; FAIL permuted_features: max_abs=2.673e-01 max_rel=8.786e+04 nl2=1.146e+00 fails=26364/26368 |
| missing_bias | **passed** |  |  |  |  | zeroed first bias flagged: true; FAIL missing_bias: max_abs=9.526e-03 max_rel=6.533e+03 nl2=4.084e-02 fails=26310/26368 |
| wrong_loss_denominator | **passed** |  |  |  |  | wrong reduction flagged: true; FAIL wrong_loss_denominator: max_abs=4.105e+02 max_rel=4.110e+02 nl2=4.110e+02 fails=1/1 |
| detached_input_gradient | **passed** |  |  |  |  | zero input gradient flagged: true; FAIL detached_input_gradient: max_abs=1.575e-05 max_rel=1.000e+00 nl2=1.000e+00 fails=20884/26880 (relative_to_max) |
| incomplete_mapping_raises | **passed** |  |  |  |  | dropping one mapping entry raises: true |

## acceptance

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| required_capabilities | **failed** |  |  |  |  | missing: String[]; not passed: ["training / adam_params_after"]; failed evidence: 2 |

