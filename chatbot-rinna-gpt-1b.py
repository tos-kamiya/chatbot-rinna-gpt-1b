import readline  # input関数の挙動を修正するため(参考: https://teratail.com/questions/157674)

import torch
from transformers import T5Tokenizer, AutoModelForCausalLM

tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt-1b")
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt-1b")

# グラフィックスカードのメモリが不足して実行できないときは次の２行をコメントアウトする
if torch.cuda.is_available():
    model = model.to("cuda")

context_header = ["人間と聡明なAIアシスタントのチャット"]

session = [
    "人間: こんにちは。あなたは何ができるの？",
    "アシスタント: AIアシスタントとして、質問に答えたりチャットすることができます。",
    "人間: 世界で一番高い山は？",
    "アシスタント: エベレスト山。",
]

H = "人間:"
A = "アシスタント:"

print('\n'.join(context_header + session))
while True:
    inp = input(H + " ")
    if not inp:
        print("exit...")
        break

    inp = inp.rstrip()
    human_said = H + " " + inp
    prompt = '\n'.join(context_header + session + [human_said])
    token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

    for t in range(5):
        with torch.no_grad():
            output_ids = model.generate(
                token_ids.to(model.device),
                max_length=150,
                min_length=100,
                do_sample=True,
                top_k=500,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        output = tokenizer.decode(output_ids.tolist()[0]).rstrip()
        if output.endswith('</s>'):
            output = output[:len('</s>')]

        generated_part = output[len(prompt):]
        p = generated_part.find(A)
        if p >= 0:
            generated_part = generated_part[p:]
            break  # for t
    else:
        print("fail to generate answer...")
        continue  # while True

    p = generated_part.find(H)
    if p >= 0:
        generated_part = generated_part[:p]
    assistant_said = generated_part
    print(assistant_said)

    session = session[2:] + [human_said, assistant_said]

