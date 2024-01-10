# trad-chinese-reading-comprehension-test-for-llms

因大語言模型的空間龐大，因此此次報告未將模型部分上傳至github。

# 作法說明
## 1. 環境建立
使用Chinese-LLaMA-Alpaca-2，並且利用conda建立其虛擬環境。
```
git clone https://github.com/ymcui/Chinese-LLaMA-Alpaca-2
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh
(重開終端機)
conda create --name llm python=3.10
conda activate llm
cd Chinese-LLaMA-Alpaca-2
pip install -r requirement.txt
```
## 2.資料處理
將資料的形式從原來的xlsx，經過處理，轉換成一json檔。  
並且處理的過程中，要將每一筆資料轉換為instruction(指令), input(指令說明), output(你預期的輸出)的形式。
```
{
    "instruction": "請根據以下輸入回答選擇題，並以數字回答：\n",
    "input": "題目",
    "output": "答案"
}
```

並且將訓練資料以8:2的方式拆為訓練集與驗證集
```python
json_data = []


for k in range(data.shape[0]):

    data_dict = {
        "instruction": "請根據以下輸入回答選擇題，並以數字回答：\n",
        "input": "",
        "output": ""
    }

    input = list(data.iloc[k][1:7])
    for i in range(1, 5):
        input[1+i] = f"{i}: " + str(input[1+i])

    input_s = "\n".join(input)
    data_dict["input"] = input_s
    data_dict["output"] = str(data.iloc[k][7])

    json_data.append(data_dict)

n = data.shape[0]
train_json, val_json = train_test_split(
    json_data,
    test_size=0.2,
    random_state=42,
)

import json
with open('train.json', 'w') as json_file:
    json.dump(train_json, json_file, indent=2)

with open('valid.json', 'w') as json_file:
    json.dump(val_json, json_file, indent=2)
```

## 3. Model
### Chinese-Alpaca-2-7B
本次閱讀選擇題回答所使用的模型為大語言模型中的Chinese-Alpaca-2-7B，礙於資源設備的關係，使用7B的模型。  
利用以下網址下載訓練模型。  
https://drive.google.com/drive/folders/1JsJDVs7tE2y31PBNleBlDPsB7S0ZrY8d  

### 模型訓練的設定
1. 將 Chinese-LLaMA-Alpaca-2/scripts/training/run_clm_sft_with_peft.py ，的340行進行註解。
```python
    #if (len(tokenizer)) != 55296:
     #   raise ValueError(f"The vocab size of the tokenizer should be 55296, but found {len(tokenizer)}.\n"
     #                    "Please use Chinese-LLaMA-2 tokenizer.")
```

2. 超參數調整

針對Chinese-LLaMA-Alpaca-2/scripts/training/run_sft.sh，進行超參數修改，修改為以下設定。  
因本身資源的顯卡記憶體大小只有8Ｇ，為了可以將模型訓練起來，超參數部分調整較小，犧牲了精確度。
- lora_rank = 8
- lora_alpha = 16
- max_seq_length=128
- "modules_to_save": null
- "load_in_kbits": 4


### 模型訓練與合成
1. 訓練
```
將以上設定完成後，即可執行run_sft.sh檔。  
將目錄移至 Chinese-LLaMA-Alpaca-2/scripts/training，執行 ./run_sft.sh。
```
2. 合成
```
最後將訓練完成的lora模型，與原來的alpaca 7B的大語言模型進行合併。
執行Chinese-LLaMA-Alpaca-2/scripts/merge_llama2_with_chinese_lora_low_mem.py，進行模型的合併
```

### Inference
合併完成後，即可使用以下程式碼進行testing data的 inference。
```python
import torch
import os
import argparse
import json,csv
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    StoppingCriteria,
    BitsAndBytesConfig
)

# 訓練時System Prompt可以自訂，生成時建議與訓練的Prompt一致
# 請參考script/training/build_dataset.py 進行Prompt的調整
DEFAULT_SYSTEM_PROMPT = """請根據以下輸入回答選擇題，並以數字回答：\n"""

TEMPLATE_WITH_SYSTEM_PROMPT = (
    "[INST] <<SYS>>\n"
    "{system_prompt}\n"
    "<</SYS>>\n\n{instruction} [/INST]"
)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--base_model',
    default=None,
    type=str,
    required=True,
    help='Base model path')
parser.add_argument(
    '--gpus',
    default="0",
    type=str,
    help='If None, cuda:0 will be used. Inference using multi-cards: --gpus=0,1,... ')
parser.add_argument(
    '--load_in_8bit',
    action='store_true',
    help='Use 8 bit quantized model')
parser.add_argument(
    '--load_in_4bit',
    action='store_true',
    help='Use 4 bit quantized model')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

# Get GPU devices
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Model
model = LlamaForCausalLM.from_pretrained(
    args.base_model,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map='auto',
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        bnb_4bit_compute_dtype=torch.float16
    )
)

# Load Tokenizer
tokenizer = LlamaTokenizer.from_pretrained(args.base_model)

# Do inference
with open('AI1000.json', 'r') as file:
    json_data = json.load(file)
    with open('answer.csv', 'w', newline='', encoding='utf-8') as csv_file:
        writer=csv.writer(csv_file)
        writer.writerow(['ID', 'Answer'])
        for row in json_data:
            id = row['id']
            instruction = row['instruction'] + '\n' + row['input']

            prompt = TEMPLATE_WITH_SYSTEM_PROMPT.format_map({'instruction': instruction,'system_prompt': DEFAULT_SYSTEM_PROMPT})
            inputs = tokenizer.encode(prompt+'\n', return_tensors="pt").to(DEV)

            generate_kwargs = dict(
                input_ids=inputs,
                temperature=0.2,
                top_p=0.9,
                top_k=40,
                do_sample=True,
                max_new_tokens=1, #為了回答選擇題而設定1
                repetition_penalty=1.1,
                guidance_scale=1.0
            )
            outputs = model.generate(**generate_kwargs)
            result = tokenizer.decode(outputs[0])
            print(result)
            response = result.split('[/INST]\n')[-1]
            writer.writerow([id, response[0]])
```

# 改進與心得討論
## 改進

1. 最後得到的準確率大約在70%，認為是超參數設定太小的原因，導致精度的效果較差。因為受限於設備資源限制，顯示卡的記憶體只有8GB，為了將大語言模型訓練成功，因此參數設定較小。未來資源充足的情況下，可以將參數設定較大，進行實驗上的對比。
2. lora_trainable的超參數設定部分，可以設定 q_proj,v_proj 進行訓練，因為有論文指出只訓練這兩者跟訓練全部的效果相差不大，藉此還可以節省資源空間。


## 心得討論
此次kaggle競賽讓我學習到大語言模型的使用方式，利用訓練lora的部分，不需要fine tune整個大語言模型，透過類似adapter的方式，即可讓大語言模型對於特定任務表現得更加完善，不過這次作業也讓我了解到，大語言模型訓練的過程中，所需要的資源量真的是非常龐大。