# Torchlight column MLP validation: column_mlp_tiny_float64.h5

Generated 2026-09-10T15:45:50.

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
- `dtype`: float64
- `fixture`: column_mlp_tiny_float64.h5
- `julia`: 1.12.7
- `n_parameters`: 166
- `source_framework`: torch
- `source_platform`: macOS-26.6.2-arm64-arm-64bit-Mach-O
- `source_version`: 2.14.0
- `widths`: [7, 11, 5, 3]

## Timings (seconds)

| step | seconds |
|---|---|
| forward_plain_first | 0.130 |
| forward_plain_warm | 0.000 |
| forward_reactant_compile | 1.133 |
| forward_reactant_run_sync_first | 0.020 |
| forward_reactant_run_sync_warm | 0.000 |
| grad_enzyme_first_total | 0.064 |
| grad_enzyme_warm | 0.000 |
| grad_reactant_compile | 13.115 |
| grad_reactant_first_total | 13.566 |
| grad_reactant_run_sync_first | 0.200 |
| grad_reactant_run_sync_warm | 0.000 |
| grad_zygote_first_total | 2.647 |
| grad_zygote_warm | 0.000 |
| setup_and_mapping | 0.955 |
| train_step_reactant_first_total | 25.157 |

## mapping

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| coverage | **passed** |  |  |  |  | MappingReport(6/6 source keys -> 6 leaves; unused=0, unmapped=0, duplicates=0, shape_mismatch=0, dtype_mismatch=0) |

## forward

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| output | **passed** | 8.327e-17 | 2.740e-16 | 0/39 | (1.000e-10, 1.000e-08) |  |
| intermediates | **passed** | 4.441e-16 | 2.761e-16 | 0/455 | (1.000e-10, 1.000e-08) | all 5 stages match |
| sensitivity_input | **passed** |  |  |  |  | max output change 0.002361465014002903 |
| sensitivity_parameters | **passed** |  |  |  |  | max output change 0.045140937361889566 |

## forward/stages

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| layer0_act | **passed** | 2.220e-16 | 1.330e-16 | 0/143 | (1.000e-10, 1.000e-08) |  |
| layer0_preact | **passed** | 4.441e-16 | 1.522e-16 | 0/143 | (1.000e-10, 1.000e-08) |  |
| layer1_act | **passed** | 1.110e-16 | 1.680e-16 | 0/65 | (1.000e-10, 1.000e-08) |  |
| layer1_preact | **passed** | 1.110e-16 | 1.540e-16 | 0/65 | (1.000e-10, 1.000e-08) |  |
| layer2_preact | **passed** | 1.110e-16 | 2.761e-16 | 0/39 | (1.000e-10, 1.000e-08) |  |

## domain/cases

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| basis | **passed** | 1.110e-16 | 3.968e-16 | 0/21 | (1.000e-10, 1.000e-08) | batch 7 |
| constant_minus2_b4 | **passed** | 5.551e-17 | 2.172e-16 | 0/12 | (1.000e-10, 1.000e-08) | batch 4 |
| large_1e3_b2 | **passed** | 1.110e-16 | 2.711e-16 | 0/6 | (1.000e-10, 1.000e-08) | batch 2 |
| ones_b1 | **passed** | 5.551e-17 | 2.195e-16 | 0/3 | (1.000e-10, 1.000e-08) | batch 1 |
| permuted_reference_batch | **passed** | 8.327e-17 | 2.654e-16 | 0/39 | (1.000e-10, 1.000e-08) | batch 13 |
| ramp_b3 | **passed** | 5.551e-17 | 1.767e-16 | 0/9 | (1.000e-10, 1.000e-08) | batch 3 |
| randn_b1 | **passed** | 5.551e-17 | 3.244e-16 | 0/3 | (1.000e-10, 1.000e-08) | batch 1 |
| randn_b2 | **passed** | 4.163e-17 | 9.812e-17 | 0/6 | (1.000e-10, 1.000e-08) | batch 2 |
| randn_b256 | **passed** | 1.665e-16 | 3.403e-16 | 0/768 | (1.000e-10, 1.000e-08) | batch 256 |
| randn_b32 | **passed** | 1.110e-16 | 2.883e-16 | 0/96 | (1.000e-10, 1.000e-08) | batch 32 |
| randn_b7 | **passed** | 1.110e-16 | 2.359e-16 | 0/21 | (1.000e-10, 1.000e-08) | batch 7 |
| randn_scale0.01_b8 | **passed** | 8.327e-17 | 2.788e-16 | 0/24 | (1.000e-10, 1.000e-08) | batch 8 |
| randn_scale100_b8 | **passed** | 1.110e-16 | 1.319e-16 | 0/24 | (1.000e-10, 1.000e-08) | batch 8 |
| randn_scale10_b8 | **passed** | 1.110e-16 | 1.998e-16 | 0/24 | (1.000e-10, 1.000e-08) | batch 8 |
| randn_scale1_b8 | **passed** | 1.110e-16 | 3.294e-16 | 0/24 | (1.000e-10, 1.000e-08) | batch 8 |
| reference_batch_first_half | **passed** | 1.110e-16 | 3.065e-16 | 0/18 | (1.000e-10, 1.000e-08) | batch 6 |
| reference_batch_second_half | **passed** | 8.327e-17 | 2.926e-16 | 0/21 | (1.000e-10, 1.000e-08) | batch 7 |
| zeros_b1 | **passed** | 5.551e-17 | 3.701e-16 | 0/3 | (1.000e-10, 1.000e-08) | batch 1 |

## domain

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| all_cases | **passed** | 1.665e-16 | 3.968e-16 | 0/1122 | (1.000e-10, 1.000e-08) | 18/18 input families match |
| column_independence | **passed** |  |  |  |  | PASS split_batch: max_abs=0.000e+00 max_rel=0.000e+00 nl2=0.000e+00 fails=0/39; PASS permuted_batch: max_abs=0.000e+00 max_rel=0.000e+00 nl2=0.000e+00 fails=0/39 |

## derivatives/zygote

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| mse_loss | **passed** | 1.110e-16 | 1.536e-16 | 0/1 | (1.000e-10, 1.000e-08) |  |
| grad_input | **passed** | 4.337e-18 | 2.968e-16 | 0/91 | (1.000e-10, 1.000e-08) |  |
| grad_params | **passed** | 5.551e-17 | 2.720e-16 | 0/166 | (1.000e-10, 1.000e-08) |  |
| vjp_input | **passed** | 5.551e-17 | 2.066e-16 | 0/91 | (1.000e-10, 1.000e-08) |  |
| vjp_params | **passed** | 8.882e-16 | 3.249e-16 | 0/166 | (1.000e-10, 1.000e-08) |  |

## derivatives/enzyme

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| mse_loss | **passed** | 0 | 0 | 0/1 | (1.000e-10, 1.000e-08) |  |
| grad_input | **passed** | 4.337e-18 | 2.968e-16 | 0/91 | (1.000e-10, 1.000e-08) |  |
| grad_params | **passed** | 5.551e-17 | 2.720e-16 | 0/166 | (1.000e-10, 1.000e-08) |  |
| vjp_input | **passed** | 5.551e-17 | 2.066e-16 | 0/91 | (1.000e-10, 1.000e-08) |  |
| vjp_params | **passed** | 8.882e-16 | 3.249e-16 | 0/166 | (1.000e-10, 1.000e-08) |  |

## derivatives/reactant

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| mse_loss | **passed** | 2.220e-16 | 3.073e-16 | 0/1 | (1.000e-10, 1.000e-08) |  |
| grad_input | **passed** | 2.602e-18 | 2.588e-16 | 0/91 | (1.000e-10, 1.000e-08) |  |
| grad_params | **passed** | 4.163e-17 | 2.981e-16 | 0/166 | (1.000e-10, 1.000e-08) |  |
| vjp_input | **passed** | 5.551e-17 | 2.470e-16 | 0/91 | (1.000e-10, 1.000e-08) |  |
| vjp_params | **passed** | 8.882e-16 | 4.731e-16 | 0/166 | (1.000e-10, 1.000e-08) |  |
| compiled_forward | **passed** | 1.110e-16 | 3.707e-16 | 0/39 | (1.000e-10, 1.000e-08) | PASS compiled_forward: max_abs=1.110e-16 max_rel=1.110e-14 nl2=3.707e-16 fails=0/39; perturbed-input change 0.0021868957812655743; perturbed-parameter change 0.03930592068736677 |

## derivatives

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| reactant_vs_zygote_grad_input | **passed** | 3.469e-18 | 2.861e-16 | 0/91 | (1.000e-10, 1.000e-08) |  |

## probes

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| finite_difference_input | **passed** | 3.427e-12 |  |  |  | best h=1.0e-5 fd=-0.010932419663989277 analytic=-0.01093241966056191 rel=3.1350495782382757e-10 (bar 1.0e-6) |
| finite_difference_params | **passed** | 4.558e-11 |  |  |  | best h=1.0e-7 fd=-0.36836061978462453 analytic=-0.36836061983020074 rel=1.237271494822273e-10 (bar 1.0e-6) |
| adjoint_consistency_cross | **passed** |  |  |  |  | input: <v,Ju>=0.5062913873839294 <Jᵀv,u>=0.5062913873839296 rel=4.385707725986873e-16; params: -14.49955940589968 vs -14.499559405899682 rel=1.2251109083200654e-16 (bar 1.0e-8) |
| jvp_input_enzyme_forward | **passed** | 7.910e-16 | 3.074e-15 | 0/39 | (1.000e-10, 1.000e-08) |  |

## training

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| sgd_grads | **passed** | 5.551e-17 | 2.720e-16 | 0/166 | (1.000e-10, 1.000e-08) |  |
| sgd_params_after | **passed** | 3.469e-18 | 2.696e-18 | 0/166 | (1.000e-10, 1.000e-08) | lr=0.05;  |
| sgd_update_delta | **passed** | 3.469e-18 | 3.089e-16 | 0/166 | (1.000e-10, 1.000e-08) |  |
| adam_params_after | **passed** | 5.551e-17 | 4.281e-17 | 0/166 | (1.000e-10, 1.000e-08) | lr=0.001 betas=(0.9,0.999) eps=1.0e-8;  |
| adam_update_delta | **passed** | 5.551e-17 | 7.485e-15 | 0/166 | (1.000e-10, 1.000e-08) |  |
| adam_state_exp_avg | **passed** | 3.469e-18 | 2.849e-16 | 0/166 | (1.000e-10, 1.000e-08) |  |
| adam_state_exp_avg_sq | **passed** | 2.033e-20 | 5.087e-16 | 0/166 | (1.000e-10, 1.000e-08) |  |
| adam_state | **passed** |  |  |  |  | exp_avg, exp_avg_sq compared in Lux layout; source step counters all == 1: true |
| compiled_sgd_step_reactant | **passed** | 6.939e-18 | 3.807e-18 | 0/166 | (1.000e-10, 1.000e-08) |  |

## dropout

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| masked_forward | **passed** | 1.388e-16 | 3.412e-16 | 0/39 | (1.000e-10, 1.000e-08) | shared 0/1 masks, scale 1/(1-p) |
| masked_grad_input | **passed** | 4.337e-18 | 3.138e-16 | 0/91 | (1.000e-10, 1.000e-08) |  |
| masked_grad_params | **passed** | 1.110e-16 | 4.416e-16 | 0/166 | (1.000e-10, 1.000e-08) |  |
| native_dropout_statistics | **passed** |  |  |  |  | keep fraction 0.9010086059570312 (expected 0.9 ± 0.01), scaling 1/(1-p): true |
| native_dropout_testmode_identity | **passed** |  |  |  |  | test mode is identity |
| native_stochastic_trajectory_match | **unsupported** |  |  |  |  | framework RNG streams differ by design; compare stochastic training statistically (not run here) |

## defect_detection

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| permuted_features | **passed** |  |  |  |  | injected feature permutation flagged: true; FAIL permuted_features: max_abs=2.686e-01 max_rel=1.220e+01 nl2=8.101e-01 fails=39/39 |
| missing_bias | **passed** |  |  |  |  | zeroed first bias flagged: true; FAIL missing_bias: max_abs=1.217e-01 max_rel=7.616e+00 nl2=4.956e-01 fails=39/39 |
| wrong_loss_denominator | **passed** |  |  |  |  | wrong reduction flagged: true; FAIL wrong_loss_denominator: max_abs=1.445e+00 max_rel=2.000e+00 nl2=2.000e+00 fails=1/1 |
| detached_input_gradient | **passed** |  |  |  |  | zero input gradient flagged: true; FAIL detached_input_gradient: max_abs=9.905e-03 max_rel=1.000e+00 nl2=1.000e+00 fails=91/91 |
| incomplete_mapping_raises | **passed** |  |  |  |  | dropping one mapping entry raises: true |

## acceptance

| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |
|---|---|---|---|---|---|---|
| required_capabilities | **passed** |  |  |  |  | all 28 required capabilities passed and no failures |

