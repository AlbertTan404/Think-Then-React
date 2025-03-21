#%%  prepare LLMs
import transformers
import torch


class LLM:
    insruction = \
"""
You are tasked with analyzing a narrative that describes the interactions between two individuals through their movements. Your goal is to identify whether their the two persons are doing intense exercise like boxing, fencing or fighting.
Respond "Yes." or "No." only without any other text.
"""

    query_template = \
"""
Here's the set of descriptions of the interaction:
{}
"""

    example_description_set_1 = \
"""
the two guys grip swords with the right hand. one strikes to the left twice, and the other moves the sword to the right twice.
two humans grip the swords in their right hand. the first one lunges twice to the left with the sword while the second one lunges twice to the right with their sword.
two performers wield swords in their right hands, while the first person swipes the sword twice to the left, the second slashes two times in the opposite direction.
"""

    example_response_1 = \
"""
Yes.
"""

    example_description_set_2 = \
"""
one squats down and picks up an object from the ground, as the other approaches with head down.
the first person bends down and picks up an item from the floor with both hands, while the second lowers their head and walks towards the first person.
one person crouches down and picks up an item from the ground with both hands, while the other approaches and lowers their head towards the first.
"""

    example_response_2 = \
"""
No.
"""


    def __init__(self, device, model_dir):
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_dir,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device,
            use_fast=False
        )
    
    def preprocess_lines(self, lines):
        res = []
        for line in lines:
            res.append(line.replace('his/her', 'his').replace('him/her', 'him').replace('he/she', 'he'))
        return res
    
    @torch.no_grad()
    def one_round_qa(self, lines):
        lines = self.preprocess_lines(lines)
        description_set = ''
        for line in lines:
            description_set += line.strip() + '\n'
        messages = [
            {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
            {"role": "user", "content": self.insruction + self.query_template.format(self.example_description_set_1)},
            {"role": "assistant", "content": self.example_response_1},
            {"role": "user", "content": self.query_template.format(self.example_description_set_2)},
            {"role": "assistant", "content": self.example_response_2},
            {"role": "user", "content": self.query_template.format(description_set)}
        ]

        prompt = self.pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.pipeline(
            prompt,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            pad_token_id=self.pipeline.tokenizer.eos_token_id
        )
        response = outputs[0]["generated_text"][len(prompt):].strip().split('\n')

        assert response[0].startswith('[Initiator]')
        assert response[1].startswith('[Receiver]')
        return {
            'action': [response[0][len('[Initiator]'):].strip(' \n.')],
            'reaction': [response[1][len('[Receiver]'):].strip(' \n.')]
        }

#%%  Prepare src data
from pathlib import Path

data_root_dir = Path('~/data/data/motion/interhuman').expanduser()
src_txt_path_list = (data_root_dir / 'texts').glob('*.txt')
tgt_dir = data_root_dir / 'annots' / 'short_thinkings'
tgt_dir.mkdir(exist_ok=True, parents=True)

name_lines = []
for src_txt_path in src_txt_path_list:
    with src_txt_path.open('r') as f:

        try:
            lines = f.readlines()
        except:
            continue
    name_lines.append((src_txt_path.stem, lines))

#%%  Start parallel processing
import os
import torch
import json
from concurrent.futures import ProcessPoolExecutor as PPE

MODEL_DIR = os.path.expanduser('~/data/pretrained_models/llm/Meta-Llama-3-8B-Instruct')


def single_process(device_idx, name_lines_chunk, model_dir=MODEL_DIR):
    llm = LLM(device=device_idx, model_dir=model_dir)
    res = {}
    for i, (name, lines) in enumerate(name_lines_chunk):
        if i % 100 == 0:
            print(i)
        try:
            result = llm.one_round_qa(lines)
            result['interaction'] = lines
        except Exception as e:
            print(e)
        else:
            res[name] = result
    return res

devices = [torch.device(f'cuda:{i}') for i in '1,2,5,6,7'.split(',')]
n_devices = len(devices)

name_lines_chunks = [
    name_lines[i: : n_devices] for i in range(n_devices)
]

# with PPE(max_workers=n_devices) as ppe:
#     list(ppe.map(single_process, devices, name_lines_chunks))
single_process(devices[0], name_lines_chunks[0])

#%%  check split data
if False:
    import json
    import random
    from pathlib import Path

    data_root_dir = Path('~/data/data/motion/Inter-X_Dataset').expanduser()
    src_txt_path_list = list((data_root_dir / 'texts').glob('*.txt'))
    random.shuffle(src_txt_path_list)
    tgt_dir = data_root_dir / 'texts_action_reaction'

    for src_txt_path in src_txt_path_list[:10]:
        stem = src_txt_path.stem
        with src_txt_path.open('r') as f:
            src_lines = f.readlines()
        with (tgt_dir / f'{stem}.json').open('r') as f:
            tgt_lines = json.load(f)
        
        for src, tgt in zip(src_lines, tgt_lines):
            print(f'{src.strip()}\n{tgt}\n')
        print('-----------------------------------')
# %%
