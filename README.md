# CSE151Project
## 1) How many observations?
- **Rows:** **10,868**
- **Columns:** **69**

## Columns, scales, and distributions
- **Heuristic typing:** **55 continuous**, **14 categorical** (including string/ID-like or low-cardinality numeric).  
- **Per-column summary** (dtype, variable type, scale, cardinality, missing%, plus brief notes):

| name | dtype | var_type | scale | n_unique | missing_% | notes |
| --- | --- | --- | --- | --- | --- | --- |
| asm_commands_add | int64 | continuous | numeric (interval/ratio) | 2122 | 0.0 |  μ≈724, σ≈1.57e+03 |
| asm_commands_call | float64 | continuous | numeric (interval/ratio) | 1993 | 0.0 |  μ≈959, σ≈2.89e+03 |
| asm_commands_cdq | float64 | continuous | numeric (interval/ratio) | 191 | 0.0 |  μ≈10.8, σ≈39.1 |
| asm_commands_cld | float64 | continuous | numeric (interval/ratio) | 210 | 0.0 |  μ≈200, σ≈1.57e+03 |
| asm_commands_cli | float64 | continuous | numeric (interval/ratio) | 200 | 0.0 |  μ≈16.8, σ≈184 |
| asm_commands_cmc | float64 | categorical (numeric-codes likely) | nominal | 54 | 0.0 |  top: 0.0(9403); 1.0(502); 2.0(245) |
| asm_commands_cmp | float64 | continuous | numeric (interval/ratio) | 1563 | 0.0 |  μ≈480, σ≈1.38e+03 |
| asm_commands_cwd | float64 | categorical (numeric-codes likely) | nominal | 46 | 0.0 |  top: 0.0(8761); 1.0(609); 2.0(355) |
| asm_commands_daa | float64 | continuous | numeric (interval/ratio) | 1007 | 0.0 |  μ≈221, σ≈498 |
| asm_commands_dd | int64 | continuous | numeric (interval/ratio) | 6180 | 0.0 |  μ≈1.7e+04, σ≈3.29e+04 |
| asm_commands_dec | float64 | continuous | numeric (interval/ratio) | 1495 | 0.0 |  μ≈370, σ≈926 |
| asm_commands_dw | float64 | continuous | numeric (interval/ratio) | 3143 | 0.0 |  μ≈1.63e+03, σ≈3.56e+03 |
| asm_commands_endp | float64 | continuous | numeric (interval/ratio) | 975 | 0.0 |  μ≈198, σ≈639 |
| asm_commands_faddp | float64 | categorical (numeric-codes likely) | nominal | 80 | 0.0 |  top: 0.0(9483); 2.0(425); 3.0(376) |
| asm_commands_fchs | float64 | categorical (numeric-codes likely) | nominal | 56 | 0.0 |  top: 0.0(9601); 1.0(570); 4.0(251) |
| asm_commands_fdiv | float64 | categorical (numeric-codes likely) | nominal | 92 | 0.0 |  top: 0.0(5279); 2.0(1423); 8.0(1078) |
| asm_commands_fdivr | float64 | categorical (numeric-codes likely) | nominal | 52 | 0.0 |  top: 0.0(8809); 1.0(606); 5.0(353) |
| asm_commands_fistp | float64 | categorical (numeric-codes likely) | nominal | 55 | 0.0 |  top: 0.0(9466); 1.0(738); 3.0(182) |
| asm_commands_fld | float64 | continuous | numeric (interval/ratio) | 337 | 0.0 |  μ≈37.5, σ≈406 |
| asm_commands_fstp | float64 | continuous | numeric (interval/ratio) | 327 | 0.0 |  μ≈34.4, σ≈342 |
| asm_commands_fword | float64 | continuous | numeric (interval/ratio) | 142 | 0.0 |  μ≈3.86, σ≈30.2 |
| asm_commands_fxch | float64 | continuous | numeric (interval/ratio) | 116 | 0.0 |  μ≈10.7, σ≈77.4 |
| asm_commands_imul | float64 | continuous | numeric (interval/ratio) | 595 | 0.0 |  μ≈540, σ≈3.31e+03 |
| asm_commands_in | int64 | continuous | numeric (interval/ratio) | 2314 | 0.0 |  μ≈1.2e+03, σ≈5.64e+03 |
| asm_commands_inc | float64 | continuous | numeric (interval/ratio) | 795 | 0.0 |  μ≈126, σ≈532 |
| asm_commands_ins | float64 | continuous | numeric (interval/ratio) | 353 | 0.0 |  μ≈44.6, σ≈241 |
| asm_commands_jb | float64 | continuous | numeric (interval/ratio) | 526 | 0.0 |  μ≈59.6, σ≈202 |
| asm_commands_je | float64 | continuous | numeric (interval/ratio) | 354 | 0.0 |  μ≈66.7, σ≈584 |
| asm_commands_jg | float64 | continuous | numeric (interval/ratio) | 516 | 0.0 |  μ≈49, σ≈161 |
| asm_commands_jl | float64 | continuous | numeric (interval/ratio) | 629 | 0.0 |  μ≈76.5, σ≈254 |
| asm_commands_jmp | float64 | continuous | numeric (interval/ratio) | 1351 | 0.0 |  μ≈321, σ≈930 |
| asm_commands_jnb | float64 | continuous | numeric (interval/ratio) | 307 | 0.0 |  μ≈23.7, σ≈90.3 |
| asm_commands_jno | float64 | categorical (numeric-codes likely) | nominal | 51 | 0.0 |  top: 0.0(9717); 1.0(408); 2.0(196) |
| asm_commands_jo | float64 | continuous | numeric (interval/ratio) | 115 | 0.0 |  μ≈6.44, σ≈67.1 |
| asm_commands_jz | float64 | continuous | numeric (interval/ratio) | 1268 | 0.0 |  μ≈343, σ≈1.16e+03 |
| asm_commands_lea | float64 | continuous | numeric (interval/ratio) | 1479 | 0.0 |  μ≈491, σ≈1.63e+03 |
| asm_commands_mov | float64 | continuous | numeric (interval/ratio) | 3805 | 0.0 |  μ≈4.22e+03, σ≈1.16e+04 |
| asm_commands_mul | float64 | continuous | numeric (interval/ratio) | 655 | 0.0 |  μ≈571, σ≈3.42e+03 |
| asm_commands_not | float64 | continuous | numeric (interval/ratio) | 329 | 0.0 |  μ≈45.1, σ≈332 |
| asm_commands_or | int64 | continuous | numeric (interval/ratio) | 4161 | 0.0 |  μ≈3.08e+03, σ≈7.67e+03 |
| asm_commands_out | float64 | continuous | numeric (interval/ratio) | 258 | 0.0 |  μ≈23.6, σ≈138 |
| asm_commands_outs | float64 | categorical (numeric-codes likely) | nominal | 78 | 0.0 |  top: 0.0(9116); 1.0(513); 2.0(288) |
| asm_commands_pop | float64 | continuous | numeric (interval/ratio) | 1643 | 0.0 |  μ≈583, σ≈1.58e+03 |
| asm_commands_push | float64 | continuous | numeric (interval/ratio) | 2645 | 0.0 |  μ≈1.77e+03, σ≈5.34e+03 |
| asm_commands_rcl | float64 | continuous | numeric (interval/ratio) | 121 | 0.0 |  μ≈4.97, σ≈23.8 |
| asm_commands_rcr | float64 | categorical (numeric-codes likely) | nominal | 93 | 0.0 |  top: 0.0(5547); 5.0(1389); 3.0(1069) |
| asm_commands_rep | float64 | continuous | numeric (interval/ratio) | 328 | 0.0 |  μ≈32.7, σ≈128 |
| asm_commands_ret | float64 | continuous | numeric (interval/ratio) | 1238 | 0.0 |  μ≈292, σ≈805 |
| asm_commands_rol | float64 | continuous | numeric (interval/ratio) | 291 | 0.0 |  μ≈38.3, σ≈457 |
| asm_commands_ror | float64 | continuous | numeric (interval/ratio) | 338 | 0.0 |  μ≈44.7, σ≈252 |
| asm_commands_sal | float64 | categorical (numeric-codes likely) | nominal | 107 | 0.0 |  top: 0.0(5606); 8.0(995); 12.0(680) |
| asm_commands_sar | float64 | continuous | numeric (interval/ratio) | 297 | 0.0 |  μ≈29.7, σ≈206 |
| asm_commands_sbb | float64 | continuous | numeric (interval/ratio) | 318 | 0.0 |  μ≈26.7, σ≈97.6 |
| asm_commands_scas | float64 | categorical (numeric-codes likely) | nominal | 82 | 0.0 |  top: 0.0(8412); 1.0(679); 2.0(468) |
| asm_commands_shl | float64 | continuous | numeric (interval/ratio) | 442 | 0.0 |  μ≈55.8, σ≈401 |
| asm_commands_shr | float64 | continuous | numeric (interval/ratio) | 422 | 0.0 |  μ≈42.5, σ≈253 |
| asm_commands_sidt | float64 | categorical (numeric-codes likely) | nominal | 18 | 0.0 |  top: 0.0(10544); 3.0(134); 4.0(73) |
| asm_commands_stc | float64 | continuous | numeric (interval/ratio) | 115 | 0.0 |  μ≈9.15, σ≈151 |
| asm_commands_std | float64 | continuous | numeric (interval/ratio) | 855 | 0.0 |  μ≈606, σ≈3.9e+03 |
| asm_commands_sti | float64 | continuous | numeric (interval/ratio) | 148 | 0.0 |  μ≈6.15, σ≈36.8 |
| asm_commands_stos | float64 | continuous | numeric (interval/ratio) | 186 | 0.0 |  μ≈15.1, σ≈55.1 |
| asm_commands_sub | float64 | continuous | numeric (interval/ratio) | 3002 | 0.0 |  μ≈2.16e+03, σ≈6.67e+03 |
| asm_commands_test | float64 | continuous | numeric (interval/ratio) | 1311 | 0.0 |  μ≈331, σ≈1.11e+03 |
| asm_commands_wait | float64 | continuous | numeric (interval/ratio) | 125 | 0.0 |  μ≈6.23, σ≈17 |
| asm_commands_xchg | float64 | continuous | numeric (interval/ratio) | 196 | 0.0 |  μ≈71.7, σ≈581 |
| asm_commands_xor | float64 | continuous | numeric (interval/ratio) | 1309 | 0.0 |  μ≈493, σ≈2.47e+03 |
| line_count_asm | int64 | continuous | numeric (interval/ratio) | 911 | 0.0 |  μ≈8.07e+04, σ≈6.45e+04 |
| size_asm | int64 | continuous | numeric (interval/ratio) | 990 | 0.0 |  μ≈4.68e+06, σ≈3.74e+06 |
| Class | int64 | categorical (numeric-codes likely) | nominal | 9 | 0.0 |  top: 3(2942); 2(2478); 1(1541) |


## Missing and duplicate values
- **Missing cells:** **0** (rows with any missing: **0**)  
- **Duplicate rows:** **80**

## 4) Target column and labels
- **Target candidate:** `Class` (multi-class).  
- **Label → family mapping**
- **1**: Ramnit
- **2**: Lollipop
- **3**: Kelihos_ver3
- **4**: Vundo
- **5**: Simda
- **6**: Tracur
- **7**: Kelihos_ver1
- **8**: Obfuscator.ACY
- **9**: Gatak

### Class distribution
| Label | Family | Count | Percent |
|------:|--------|------:|--------:|
| 1 | Ramnit | 1,541 | 14.18% |
| 2 | Lollipop | 2,478 | 22.80% |
| 3 | Kelihos_ver3 | 2,942 | 27.07% |
| 4 | Vundo | 475 | 4.37% |
| 5 | Simda | 42 | 0.39% |
| 6 | Tracur | 751 | 6.91% |
| 7 | Kelihos_ver1 | 398 | 3.66% |
| 8 | Obfuscator.ACY | 1,228 | 11.30% |
| 9 | Gatak | 1,013 | 9.32% |

![Class distribution](https://github.com/zonglinz/CSE151Project/blob/main/class_distribution.png?raw=true)
