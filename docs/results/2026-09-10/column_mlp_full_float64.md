# Torchlight column MLP validation: column_mlp_full_float64.h5

Generated 2026-09-10T15:44:15.

| passed | failed | unsupported | not_tested |
|---|---|---|---|
| 75 | 0 | 1 | 0 |

## Environment

- `Enzyme`: 0.13.202
- `Lux`: 1.31.4
- `Optimisers`: 0.4.9
- `Reactant`: 0.2.285
- `Zygote`: 0.7.13
- `blas_threads`: 6
- `dtype`: float64
- `fixture`: column_mlp_full_float64.h5
- `julia`: 1.12.7
- `n_parameters`: 1214876
- `source_framework`: torch
- `source_platform`: macOS-26.6.2-arm64-arm-64bit-Mach-O
- `source_version`: 2.14.0
- `widths`: [420, 512, 512, 512, 512, 412]

## Timings (seconds)

| step | seconds |
|---|---|
| forward_plain_first | 0.111 |
| forward_plain_warm | 0.001 |
| forward_reactant_compile | 1.265 |
| forward_reactant_run_sync_first | 0.029 |
| forward_reactant_run_sync_warm | 0.001 |
| grad_enzyme_first_total | 0.088 |
| grad_enzyme_warm | 0.008 |
| grad_reactant_compile | 13.128 |
| grad_reactant_first_total | 13.615 |
| grad_reactant_run_sync_first | 0.223 |
| grad_reactant_run_sync_warm | 0.002 |
| grad_zygote_first_total | 4.754 |
| grad_zygote_warm | 0.003 |
| setup_and_mapping | 0.965 |
| train_step_reactant_first_total | 25.584 |

## mapping

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| coverage | **passed** |  |  |  |  | MappingReport(10/10 source keys -> 10 leaves; unused=0, unmapped=0, duplicates=0, shape_mismatch=0, dtype_mismatch=0) |

## forward

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| output | **passed** | 6.106e-16 | 1.693e-15 | 0/26368 | (1.000e-10, 1.000e-08) |  |
| intermediates | **passed** | 3.775e-15 | 1.667e-15 | 0/288512 | (1.000e-10, 1.000e-08) | all 9 stages match |
| sensitivity_input | **passed** |  |  |  |  | max output change 0.0019364699184048761 |
| sensitivity_parameters | **passed** |  |  |  |  | max output change 0.2168285436188052 |

## forward/stages

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| layer0_act | **passed** | 1.776e-15 | 5.690e-16 | 0/32768 | (1.000e-10, 1.000e-08) |  |
| layer0_preact | **passed** | 3.775e-15 | 6.894e-16 | 0/32768 | (1.000e-10, 1.000e-08) |  |
| layer1_act | **passed** | 1.554e-15 | 9.867e-16 | 0/32768 | (1.000e-10, 1.000e-08) |  |
| layer1_preact | **passed** | 2.109e-15 | 1.026e-15 | 0/32768 | (1.000e-10, 1.000e-08) |  |
| layer2_act | **passed** | 1.166e-15 | 1.315e-15 | 0/32768 | (1.000e-10, 1.000e-08) |  |
| layer2_preact | **passed** | 1.554e-15 | 1.328e-15 | 0/32768 | (1.000e-10, 1.000e-08) |  |
| layer3_act | **passed** | 7.216e-16 | 1.554e-15 | 0/32768 | (1.000e-10, 1.000e-08) |  |
| layer3_preact | **passed** | 7.772e-16 | 1.557e-15 | 0/32768 | (1.000e-10, 1.000e-08) |  |
| layer4_preact | **passed** | 4.580e-16 | 1.667e-15 | 0/26368 | (1.000e-10, 1.000e-08) |  |

## domain/cases

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| basis | **passed** | 3.192e-16 | 1.250e-15 | 0/173040 | (1.000e-10, 1.000e-08) | batch 420 |
| constant_minus2_b4 | **passed** | 4.441e-16 | 1.638e-15 | 0/1648 | (1.000e-10, 1.000e-08) | batch 4 |
| large_1e3_b2 | **passed** | 1.027e-15 | 3.477e-15 | 0/824 | (1.000e-10, 1.000e-08) | batch 2 |
| ones_b1 | **passed** | 2.776e-16 | 1.399e-15 | 0/412 | (1.000e-10, 1.000e-08) | batch 1 |
| permuted_reference_batch | **passed** | 6.106e-16 | 1.693e-15 | 0/26368 | (1.000e-10, 1.000e-08) | batch 64 |
| ramp_b3 | **passed** | 3.053e-16 | 1.616e-15 | 0/1236 | (1.000e-10, 1.000e-08) | batch 3 |
| randn_b1 | **passed** | 2.706e-16 | 1.424e-15 | 0/412 | (1.000e-10, 1.000e-08) | batch 1 |
| randn_b2 | **passed** | 3.400e-16 | 1.730e-15 | 0/824 | (1.000e-10, 1.000e-08) | batch 2 |
| randn_b256 | **passed** | 5.829e-16 | 1.689e-15 | 0/105472 | (1.000e-10, 1.000e-08) | batch 256 |
| randn_b32 | **passed** | 4.441e-16 | 1.709e-15 | 0/13184 | (1.000e-10, 1.000e-08) | batch 32 |
| randn_b7 | **passed** | 3.608e-16 | 1.693e-15 | 0/2884 | (1.000e-10, 1.000e-08) | batch 7 |
| randn_scale0.01_b8 | **passed** | 2.082e-16 | 1.224e-15 | 0/3296 | (1.000e-10, 1.000e-08) | batch 8 |
| randn_scale100_b8 | **passed** | 8.604e-16 | 2.637e-15 | 0/3296 | (1.000e-10, 1.000e-08) | batch 8 |
| randn_scale10_b8 | **passed** | 5.551e-16 | 1.718e-15 | 0/3296 | (1.000e-10, 1.000e-08) | batch 8 |
| randn_scale1_b8 | **passed** | 4.163e-16 | 1.698e-15 | 0/3296 | (1.000e-10, 1.000e-08) | batch 8 |
| reference_batch_first_half | **passed** | 4.441e-16 | 1.685e-15 | 0/13184 | (1.000e-10, 1.000e-08) | batch 32 |
| reference_batch_second_half | **passed** | 6.106e-16 | 1.702e-15 | 0/13184 | (1.000e-10, 1.000e-08) | batch 32 |
| zeros_b1 | **passed** | 2.082e-16 | 1.047e-15 | 0/412 | (1.000e-10, 1.000e-08) | batch 1 |

## domain

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| all_cases | **passed** | 1.027e-15 | 3.477e-15 | 0/366268 | (1.000e-10, 1.000e-08) | 18/18 input families match |
| column_independence | **passed** |  |  |  |  | PASS split_batch: max_abs=0.000e+00 max_rel=0.000e+00 nl2=0.000e+00 fails=0/26368; PASS permuted_batch: max_abs=0.000e+00 max_rel=0.000e+00 nl2=0.000e+00 fails=0/26368 |

## derivatives/zygote

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| mse_loss | **passed** | 0 | 0 | 0/1 | (1.000e-10, 1.000e-08) |  |
| grad_input | **passed** | 3.388e-20 | 1.907e-15 | 0/26880 | (1.000e-10, 1.000e-08) |  |
| grad_params | **passed** | 1.301e-18 | 1.753e-15 | 0/1214876 | (1.000e-10, 1.000e-08) |  |
| vjp_input | **passed** | 4.996e-16 | 1.892e-15 | 0/26880 | (1.000e-10, 1.000e-08) |  |
| vjp_params | **passed** | 1.155e-14 | 1.933e-15 | 0/1214876 | (1.000e-10, 1.000e-08) |  |

## derivatives/enzyme

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| mse_loss | **passed** | 2.220e-16 | 2.203e-16 | 0/1 | (1.000e-10, 1.000e-08) |  |
| grad_input | **passed** | 3.388e-20 | 1.907e-15 | 0/26880 | (1.000e-10, 1.000e-08) |  |
| grad_params | **passed** | 1.301e-18 | 1.753e-15 | 0/1214876 | (1.000e-10, 1.000e-08) |  |
| vjp_input | **passed** | 4.996e-16 | 1.892e-15 | 0/26880 | (1.000e-10, 1.000e-08) |  |
| vjp_params | **passed** | 1.155e-14 | 1.933e-15 | 0/1214876 | (1.000e-10, 1.000e-08) |  |

## derivatives/reactant

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| mse_loss | **passed** | 0 | 0 | 0/1 | (1.000e-10, 1.000e-08) |  |
| grad_input | **passed** | 3.727e-20 | 1.615e-15 | 0/26880 | (1.000e-10, 1.000e-08) |  |
| grad_params | **passed** | 4.879e-19 | 1.561e-15 | 0/1214876 | (1.000e-10, 1.000e-08) |  |
| vjp_input | **passed** | 3.053e-16 | 1.225e-15 | 0/26880 | (1.000e-10, 1.000e-08) |  |
| vjp_params | **passed** | 1.066e-14 | 1.566e-15 | 0/1214876 | (1.000e-10, 1.000e-08) |  |
| compiled_forward | **passed** | 5.551e-16 | 1.638e-15 | 0/26368 | (1.000e-10, 1.000e-08) | PASS compiled_forward: max_abs=5.551e-16 max_rel=1.055e-11 nl2=1.638e-15 fails=0/26368; perturbed-input change 0.002131905253657966; perturbed-parameter change 0.2080970684972448 |

## derivatives

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| reactant_vs_zygote_grad_input | **passed** | 3.134e-20 | 1.809e-15 | 0/26880 | (1.000e-10, 1.000e-08) |  |

## probes

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| finite_difference_input | **passed** | 8.333e-13 |  |  |  | best h=1.0e-5 fd=0.0006614416014905089 analytic=0.0006614416023237599 rel=1.259749915927842e-9 (bar 1.0e-6) |
| finite_difference_params | **passed** | 1.996e-10 |  |  |  | best h=1.0e-6 fd=-0.03807129000943377 analytic=-0.03807128980978758 rel=5.244009054266399e-9 (bar 1.0e-6) |
| adjoint_consistency_cross | **passed** |  |  |  |  | input: <v,Ju>=0.9509785427032549 <Jᵀv,u>=0.9509785427032318 rel=2.4283028349472578e-14; params: -576.1304958977006 vs -576.1304958976968 rel=6.511833120320514e-15 (bar 1.0e-8) |
| jvp_input_enzyme_forward | **passed** | 6.245e-16 | 3.149e-15 | 0/26368 | (1.000e-10, 1.000e-08) |  |

## training

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| sgd_grads | **passed** | 1.301e-18 | 1.753e-15 | 0/1214876 | (1.000e-10, 1.000e-08) |  |
| sgd_params_after | **passed** | 6.939e-18 | 9.527e-18 | 0/1214876 | (1.000e-10, 1.000e-08) | lr=0.05;  |
| sgd_update_delta | **passed** | 6.939e-18 | 4.834e-14 | 0/1214876 | (1.000e-10, 1.000e-08) |  |
| adam_params_after | **passed** | 1.708e-14 | 2.968e-15 | 0/1214876 | (1.000e-10, 1.000e-08) | lr=0.001 betas=(0.9,0.999) eps=1.0e-8;  |
| adam_update_delta | **passed** | 1.708e-14 | 7.592e-14 | 0/1214876 | (1.000e-10, 1.000e-08) |  |
| adam_state_exp_avg | **passed** | 1.220e-19 | 1.755e-15 | 0/1214876 | (1.000e-10, 1.000e-08) |  |
| adam_state_exp_avg_sq | **passed** | 2.895e-24 | 2.074e-15 | 0/1214876 | (1.000e-10, 1.000e-08) |  |
| adam_state | **passed** |  |  |  |  | exp_avg, exp_avg_sq compared in Lux layout; source step counters all == 1: true |
| compiled_sgd_step_reactant | **passed** | 6.939e-18 | 9.346e-18 | 0/1214876 | (1.000e-10, 1.000e-08) |  |

## dropout

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| masked_forward | **passed** | 6.245e-16 | 1.665e-15 | 0/26368 | (1.000e-10, 1.000e-08) | shared 0/1 masks, scale 1/(1-p) |
| masked_grad_input | **passed** | 4.405e-20 | 1.831e-15 | 0/26880 | (1.000e-10, 1.000e-08) |  |
| masked_grad_params | **passed** | 9.758e-19 | 1.694e-15 | 0/1214876 | (1.000e-10, 1.000e-08) |  |
| native_dropout_statistics | **passed** |  |  |  |  | keep fraction 0.9010086059570312 (expected 0.9 ± 0.01), scaling 1/(1-p): true |
| native_dropout_testmode_identity | **passed** |  |  |  |  | test mode is identity |
| native_stochastic_trajectory_match | **unsupported** |  |  |  |  | framework RNG streams differ by design; compare stochastic training statistically (not run here) |

## defect_detection

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| permuted_features | **passed** |  |  |  |  | injected feature permutation flagged: true; FAIL permuted_features: max_abs=2.596e-01 max_rel=1.371e+04 nl2=1.153e+00 fails=26368/26368 |
| missing_bias | **passed** |  |  |  |  | zeroed first bias flagged: true; FAIL missing_bias: max_abs=9.509e-03 max_rel=4.321e+02 nl2=4.101e-02 fails=26368/26368 |
| wrong_loss_denominator | **passed** |  |  |  |  | wrong reduction flagged: true; FAIL wrong_loss_denominator: max_abs=4.143e+02 max_rel=4.110e+02 nl2=4.110e+02 fails=1/1 |
| detached_input_gradient | **passed** |  |  |  |  | zero input gradient flagged: true; FAIL detached_input_gradient: max_abs=1.701e-05 max_rel=1.000e+00 nl2=1.000e+00 fails=26880/26880 |
| incomplete_mapping_raises | **passed** |  |  |  |  | dropping one mapping entry raises: true |

## acceptance

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| required_capabilities | **passed** |  |  |  |  | all 28 required capabilities passed and no failures |

