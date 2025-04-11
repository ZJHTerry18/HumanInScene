import json
from openai import OpenAI
from tqdm import tqdm
import numpy as np
import argparse
import re
import os
from collections import defaultdict
import pyrootutils
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

my_api_key = "your_api_key"
client = OpenAI(api_key=my_api_key)

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rf', type=str)
    parser.add_argument('--n_r', type=int, default=3)
    parser.add_argument('--output', type=str, default='evaluation/eval')
    args = parser.parse_args()
    return args

def main(args):
    if args.rf.endswith('jsonl'):
        pred_results = []
        with open(args.rf, 'r') as f:
            for line in f:
                pred_results.append(json.loads(line))
    else:
        with open(args.rf, 'r') as f:
            pred_results = json.load(f)
    
    print("Evaluating answers with GPT...")
    outputs = []
    for n in range(args.n_r):
        for pr in tqdm(pred_results):
            grades = []
            for qa in pr["qa"]:
                question = qa["question"]
                gt_ans = qa["gt"] if isinstance(qa["gt"], str) else qa["gt"][0]
                ans = qa["pred"]

    #             system_prompt = f'''
    # [Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant's answer. Begin your evaluation by comparing the assistant's answer with the reference answer. Identify and correct any mistakes. The assistant has access to a video alongwith questions but you will not be given videos. Therefore, please consider only how the answer is close to the reference answer. If the assistant's answer is not exactly same as or similar to the answer, then he must be wrong. Be as objective as possible. Discourage uninformative answers. Also, equally treat short and long answers and focus on the correctness of answers. Rate the response with either 0, 0.5 or 1.\n\n[Question]\n{question}\n\n[The Start of Reference Answer]\n{gt_ans}\n[The End of Reference Answer]\n\n[The Start of Assistant's Answer]\n{ans}\n[The End of Assistant's Answer]
    # Use the following format:
    # Evaluation: your evaluations.
    # Rating: the rating you give, answer with only a float number, either 0, 0.5 or 1.
    #             '''

                system_prompt = f'''
[Instruction]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant's answer. Begin your evaluation by comparing the assistant's answer with the reference answer. If the assistant's answer is not exactly same as or similar to the answer, then he must be wrong. Be as objective as possible. Discourage uninformative answers. Also, equally treat short and long answers and focus on the correctness of answers. Rate the response with either 0 (wrong), 0.5 (partially correct), 1 (almost correct) or 2 (totally correct).

[Question]
{question}

[The Start of Reference Answer]
{gt_ans}
[The End of Reference Answer]

[The Start of Assistant's Answer]
{ans}
[The End of Assistant's Answer]


Use the following format:
Evaluation: your evaluations.
Rating: the rating you give, answer with only a float number, either 0, 0.5, 1 or 2.
                '''

                messages = [{"role": "system", "content": system_prompt}]
                try:
                    response = client.chat.completions.create(
                        model='gpt-4o',
                        messages=messages,
                        max_tokens=1000,
                        temperature=0.0,
                    )

                    grade_string = response.choices[0].message.content
                    grade = re.search(r"Rating:(.*)", grade_string, re.DOTALL).group(1).strip()
                except Exception as e:
                    print(f"Error for question: {e}, set grade as 0.0")
                    grade = 0
                grade = float(grade)
                grades.append(grade)

            pr["grade"] = np.mean(grade)
            # print(pr)
            outputs.append(pr)
    
    # calculate score
    print("Calculating Scores...")
    task_score = defaultdict(list)
    for od in outputs:
        task = od["task"]
        score = od["grade"]
        task_score[task].append(score)
    for task, scores in task_score.items():
        task_score[task] = np.sum(scores) / args.n_r
        print(f"{task}: {task_score[task]}")

    # save outputs
    print("Saving results...")
    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, os.path.basename(args.rf).rstrip('l')) # change .jsonl to .json
    output_content = dict()
    output_content["scores"] = task_score
    output_content["res"] = outputs
    with open(output_path, 'w') as f:
        json.dump(output_content, f)

if __name__ == "__main__":
    args = argparser()
    main(args)