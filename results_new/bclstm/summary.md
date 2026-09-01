# bc-LSTM context ablation

93 of 30 cells complete.

## Dev weighted F1 (emotion)

| condition | k0 | k1 | k2 | k4 | full |
|---|---|---|---|---|---|
| `gold` | 0.5853 | 0.5925 | 0.5954 | 0.6109 | 0.5931 |
| `asr` | 0.5018 | 0.4984 | 0.5044 | 0.5066 | 0.4975 |
| `asr_cleaned` | 0.5003 | 0.4982 | 0.4983 | 0.5041 | 0.5054 |
| `stacked_gold` | 0.5876 | 0.5799 | 0.5837 | 0.5922 | 0.5902 |
| `stacked_asr` | 0.5122 | 0.5141 | 0.5171 | 0.5157 | 0.5111 |
| `stacked_asr_cleaned` | 0.4962 | 0.4928 | 0.4916 | 0.4932 | 0.4939 |

## Test weighted F1 (emotion)

| condition | k0 | k1 | k2 | k4 | full |
|---|---|---|---|---|---|
| `gold` | 0.6185 | 0.6258 | 0.6203 | 0.6180 | 0.6220 |
| `asr` | 0.4919 | 0.4930 | 0.4925 | 0.4935 | 0.4945 |
| `asr_cleaned` | 0.5073 | 0.5100 | 0.5128 | 0.5101 | 0.5075 |
| `stacked_gold` | 0.6197 | 0.6206 | 0.6230 | 0.6230 | 0.6274 |
| `stacked_asr` | 0.5061 | 0.5057 | 0.5011 | 0.5030 | 0.5089 |
| `stacked_asr_cleaned` | 0.5030 | 0.5049 | 0.5094 | 0.5074 | 0.5093 |

## Per-run detail

| condition | context | dev WF1 | test WF1 | best ep | epochs | early stop | params |
|---|---|---|---|---|---|---|---|
| `asr` | `k0` | 0.5018 | 0.4919 | 8 | 14 | yes | 4727818 |
| `asr` | `k1` | 0.4984 | 0.4930 | 6 | 12 | yes | 4727818 |
| `asr` | `k2` | 0.5044 | 0.4925 | 5 | 11 | yes | 4727818 |
| `asr` | `k4` | 0.5066 | 0.4935 | 5 | 11 | yes | 4727818 |
| `asr` | `full` | 0.4975 | 0.4945 | 3 | 9 | yes | 4727818 |
| `asr_cleaned` | `k0` | 0.5003 | 0.5073 | 14 | 20 | yes | 4727818 |
| `asr_cleaned` | `k1` | 0.4982 | 0.5100 | 2 | 8 | yes | 4727818 |
| `asr_cleaned` | `k2` | 0.4983 | 0.5128 | 4 | 10 | yes | 4727818 |
| `asr_cleaned` | `k4` | 0.5041 | 0.5101 | 1 | 7 | yes | 4727818 |
| `asr_cleaned` | `full` | 0.5054 | 0.5075 | 9 | 15 | yes | 4727818 |
| `attn_asr` | `k0` | 0.5186 | 0.5145 | 5 | 11 | yes | 3154954 |
| `attn_asr` | `k1` | 0.5246 | 0.5055 | 6 | 12 | yes | 3154954 |
| `attn_asr` | `k2` | 0.5277 | 0.5140 | 6 | 12 | yes | 3154954 |
| `attn_asr` | `k4` | 0.5277 | 0.5119 | 1 | 7 | yes | 3154954 |
| `attn_asr` | `full` | 0.5339 | 0.5138 | 3 | 9 | yes | 3154954 |
| `attn_asr_cleaned` | `k0` | 0.5254 | 0.5334 | 1 | 7 | yes | 3154954 |
| `attn_asr_cleaned` | `k1` | 0.5299 | 0.5259 | 1 | 7 | yes | 3154954 |
| `attn_asr_cleaned` | `k2` | 0.5269 | 0.5326 | 3 | 9 | yes | 3154954 |
| `attn_asr_cleaned` | `k4` | 0.5242 | 0.5317 | 1 | 7 | yes | 3154954 |
| `attn_asr_cleaned` | `full` | 0.5283 | 0.5243 | 3 | 9 | yes | 3154954 |
| `attn_gold` | `k0` | 0.5952 | 0.6259 | 1 | 7 | yes | 3154954 |
| `attn_gold` | `k1` | 0.5999 | 0.6283 | 2 | 8 | yes | 3154954 |
| `attn_gold` | `k2` | 0.6105 | 0.6162 | 5 | 11 | yes | 3154954 |
| `attn_gold` | `k4` | 0.6116 | 0.6182 | 5 | 11 | yes | 3154954 |
| `attn_gold` | `full` | 0.6149 | 0.6252 | 3 | 9 | yes | 3154954 |
| `attnraw_asr` | `k0` | 0.4897 | 0.4918 | 2 | 8 | yes | 4727818 |
| `attnraw_asr` | `k1` | 0.4970 | 0.4917 | 6 | 12 | yes | 4727818 |
| `attnraw_asr` | `k2` | 0.5066 | 0.4994 | 3 | 9 | yes | 4727818 |
| `attnraw_asr` | `k4` | 0.5029 | 0.5015 | 2 | 8 | yes | 4727818 |
| `attnraw_asr` | `full` | 0.5153 | 0.5008 | 4 | 10 | yes | 4727818 |
| `attnraw_asr_cleaned` | `k0` | 0.5118 | 0.5075 | 5 | 11 | yes | 4727818 |
| `attnraw_asr_cleaned` | `k1` | 0.5172 | 0.5076 | 6 | 12 | yes | 4727818 |
| `attnraw_asr_cleaned` | `k2` | 0.5131 | 0.5022 | 6 | 12 | yes | 4727818 |
| `attnraw_asr_cleaned` | `k4` | 0.5184 | 0.5095 | 6 | 12 | yes | 4727818 |
| `attnraw_asr_cleaned` | `full` | 0.5076 | 0.5120 | 5 | 11 | yes | 4727818 |
| `attnraw_gold` | `k0` | 0.5894 | 0.6207 | 5 | 11 | yes | 4727818 |
| `attnraw_gold` | `k1` | 0.5945 | 0.6206 | 2 | 8 | yes | 4727818 |
| `attnraw_gold` | `k2` | 0.6054 | 0.6250 | 2 | 8 | yes | 4727818 |
| `attnraw_gold` | `k4` | 0.6109 | 0.6263 | 4 | 10 | yes | 4727818 |
| `attnraw_gold` | `full` | 0.6157 | 0.6263 | 10 | 16 | yes | 4727818 |
| `gold` | `k0` | 0.5853 | 0.6185 | 5 | 11 | yes | 4727818 |
| `gold` | `k1` | 0.5925 | 0.6258 | 5 | 11 | yes | 4727818 |
| `gold` | `k2` | 0.5954 | 0.6203 | 2 | 8 | yes | 4727818 |
| `gold` | `k4` | 0.6109 | 0.6180 | 16 | 22 | yes | 4727818 |
| `gold` | `full` | 0.5931 | 0.6220 | 3 | 9 | yes | 4727818 |
| `hidden_asr` | `h1024` | 0.5069 | 0.4953 | 7 | 13 | yes | 25202698 |
| `hidden_asr` | `h128` | 0.4983 | 0.4977 | 15 | 21 | yes | 2232842 |
| `hidden_asr` | `h1536` | 0.5103 | 0.4877 | 7 | 13 | yes | 44095498 |
| `hidden_asr` | `h2048` | 0.5081 | 0.4851 | 7 | 13 | yes | 67182602 |
| `hidden_asr` | `h256` | 0.4975 | 0.4945 | 3 | 9 | yes | 4727818 |
| `hidden_asr` | `h512` | 0.5008 | 0.4952 | 7 | 13 | yes | 10504202 |
| `hidden_asr_cleaned` | `h1024` | 0.5031 | 0.5145 | 4 | 10 | yes | 25202698 |
| `hidden_asr_cleaned` | `h128` | 0.5016 | 0.5099 | 11 | 17 | yes | 2232842 |
| `hidden_asr_cleaned` | `h1536` | 0.5023 | 0.5140 | 4 | 10 | yes | 44095498 |
| `hidden_asr_cleaned` | `h2048` | 0.5128 | 0.4981 | 0 | 6 | yes | 67182602 |
| `hidden_asr_cleaned` | `h256` | 0.5054 | 0.5075 | 9 | 15 | yes | 4727818 |
| `hidden_asr_cleaned` | `h512` | 0.5029 | 0.5105 | 6 | 12 | yes | 10504202 |
| `hidden_gold` | `h1024` | 0.5940 | 0.6221 | 8 | 14 | yes | 25202698 |
| `hidden_gold` | `h128` | 0.5844 | 0.6161 | 3 | 9 | yes | 2232842 |
| `hidden_gold` | `h1536` | 0.5940 | 0.6271 | 4 | 10 | yes | 44095498 |
| `hidden_gold` | `h2048` | 0.5996 | 0.6267 | 4 | 10 | yes | 67182602 |
| `hidden_gold` | `h256` | 0.5931 | 0.6220 | 3 | 9 | yes | 4727818 |
| `hidden_gold` | `h512` | 0.5966 | 0.6234 | 10 | 16 | yes | 10504202 |
| `stacked_asr` | `k0` | 0.5122 | 0.5061 | 13 | 19 | yes | 1582090 |
| `stacked_asr` | `k1` | 0.5141 | 0.5057 | 11 | 17 | yes | 1582090 |
| `stacked_asr` | `k2` | 0.5171 | 0.5011 | 6 | 12 | yes | 1582090 |
| `stacked_asr` | `k4` | 0.5157 | 0.5030 | 0 | 6 | yes | 1582090 |
| `stacked_asr` | `full` | 0.5111 | 0.5089 | 3 | 9 | yes | 1582090 |
| `stacked_asr_cleaned` | `k0` | 0.4962 | 0.5030 | 0 | 6 | yes | 1582090 |
| `stacked_asr_cleaned` | `k1` | 0.4928 | 0.5049 | 0 | 6 | yes | 1582090 |
| `stacked_asr_cleaned` | `k2` | 0.4916 | 0.5094 | 0 | 6 | yes | 1582090 |
| `stacked_asr_cleaned` | `k4` | 0.4932 | 0.5074 | 1 | 7 | yes | 1582090 |
| `stacked_asr_cleaned` | `full` | 0.4939 | 0.5093 | 6 | 12 | yes | 1582090 |
| `stacked_gold` | `k0` | 0.5876 | 0.6197 | 2 | 8 | yes | 1582090 |
| `stacked_gold` | `k1` | 0.5799 | 0.6206 | 2 | 8 | yes | 1582090 |
| `stacked_gold` | `k2` | 0.5837 | 0.6230 | 2 | 8 | yes | 1582090 |
| `stacked_gold` | `k4` | 0.5922 | 0.6230 | 10 | 16 | yes | 1582090 |
| `stacked_gold` | `full` | 0.5902 | 0.6274 | 4 | 10 | yes | 1582090 |
| `stackedcat_asr` | `k0` | 0.5047 | 0.5000 | 1 | 7 | yes | 5776394 |
| `stackedcat_asr` | `k1` | 0.5049 | 0.5032 | 5 | 11 | yes | 5776394 |
| `stackedcat_asr` | `k2` | 0.5077 | 0.4971 | 5 | 11 | yes | 5776394 |
| `stackedcat_asr` | `k4` | 0.5075 | 0.4915 | 4 | 10 | yes | 5776394 |
| `stackedcat_asr` | `full` | 0.5125 | 0.5053 | 7 | 13 | yes | 5776394 |
| `stackedcat_asr_cleaned` | `k0` | 0.4923 | 0.5033 | 0 | 6 | yes | 5776394 |
| `stackedcat_asr_cleaned` | `k1` | 0.5027 | 0.5053 | 0 | 6 | yes | 5776394 |
| `stackedcat_asr_cleaned` | `k2` | 0.4935 | 0.5089 | 1 | 7 | yes | 5776394 |
| `stackedcat_asr_cleaned` | `k4` | 0.4998 | 0.5047 | 1 | 7 | yes | 5776394 |
| `stackedcat_asr_cleaned` | `full` | 0.5081 | 0.5160 | 0 | 6 | yes | 5776394 |
| `stackedcat_gold` | `k0` | 0.5827 | 0.6172 | 0 | 6 | yes | 5776394 |
| `stackedcat_gold` | `k1` | 0.5839 | 0.6253 | 2 | 8 | yes | 5776394 |
| `stackedcat_gold` | `k2` | 0.5947 | 0.6266 | 1 | 7 | yes | 5776394 |
| `stackedcat_gold` | `k4` | 0.5920 | 0.6260 | 0 | 6 | yes | 5776394 |
| `stackedcat_gold` | `full` | 0.5988 | 0.6274 | 3 | 9 | yes | 5776394 |

## Comparability

- dev key-set hashes: `['72c2797a81a4']`
- CONSISTENT — all runs scored on the same dev set
- note: `asr_cleaned` filters TRAIN labels only; dev and test keep every utterance, so all cells score the same dev set
