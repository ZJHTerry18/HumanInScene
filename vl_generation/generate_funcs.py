import json
import re

def generate_cap_allinfo(
        motion_anns, 
        scene_anns, 
        in_context_examples,
        input_template,
        system_prompt,
        client,
        gpt_model,
        **kwargs
    ):
    object_information = json.dumps(scene_anns['object information'])
    action = motion_anns['action']
    key_moments = json.dumps(motion_anns['key moments'])

    # print(object_information)
    # print(action)
    # print(key_moments)
    user_input = input_template.format(
        object_information, action, key_moments
    )

    icl_inputs = []
    for icl in in_context_examples:
        query = input_template.format(
            json.dumps(icl['object information']),
            icl['action'],
            json.dumps(icl['key moments'])
        )
        response = icl['output']
        icl_inputs.append({
            'query': query,
            'response': response
        })

    messages = [{"role": "system", "content": system_prompt}]
    for sample in icl_inputs:
        messages.append({"role": "user", "content": sample['query']})
        messages.append({"role": "assistant", "content": sample['response']})
    messages.append({"role": "user", "content": user_input})
    # for msg in messages:
    #     print(msg)
    #     print('\n')

    response = client.chat.completions.create(
        model=gpt_model,
        messages=messages,
        max_tokens=1000,
        temperature=0.0,
    )
    # print(response)
    ans = response.choices[0].message.content
    
    return ans

def generate_qa_action(
        motion_anns,  
        in_context_examples,
        input_template,
        output_template,
        system_prompt,
        client,
        gpt_model,
        **kwargs,
    ):
    action = motion_anns['action']
    
    user_input = input_template.format(
        action
    )

    icl_inputs = []
    for icl in in_context_examples:
        query = input_template.format(
            icl['action'],
        )
        response_qas = [
            output_template.format(d['question'], d['answer']) for d in icl['qa']
        ]
        response = '\n'.join(response_qas)

        icl_inputs.append({
            'query': query,
            'response': response
        })

    messages = [{"role": "system", "content": system_prompt}]
    for sample in icl_inputs:
        messages.append({"role": "user", "content": sample['query']})
        messages.append({"role": "assistant", "content": sample['response']})
    messages.append({"role": "user", "content": user_input})
    # for msg in messages:
    #     print(msg)
    #     print('\n')

    response = client.chat.completions.create(
        model=gpt_model,
        messages=messages,
        max_tokens=1000,
        temperature=0.0,
    )
    motion_scene_qas = response.choices[0].message.content
    # print(response)

    qa_outputs = re.findall(r"Question:(.*?)Answer:(.*?)(?=Question:|$)", motion_scene_qas, re.DOTALL)
    motion_scene_qa_outputs = []
    for qa in qa_outputs:
        question, answer = qa
        motion_scene_qa_outputs.append({
            "question": question.strip(),
            "answer": answer.strip()
        })

    return motion_scene_qa_outputs

def generate_qa_allinfo(
        motion_anns,
        scene_anns,  
        in_context_examples,
        input_template,
        output_template,
        system_prompt,
        client,
        gpt_model,
        **kwargs,
    ):
    object_information = json.dumps(scene_anns['object information'])
    action = motion_anns['action']
    key_moments = json.dumps(motion_anns['key moments'])
    
    user_input = input_template.format(
        object_information, action, key_moments
    )

    icl_inputs = []
    for icl in in_context_examples:
        query = input_template.format(
            json.dumps(icl['object information']),
            icl['action'],
            json.dumps(icl['key moments'])
        )
        response_qas = [
            output_template.format(d['question'], d['answer']) for d in icl['qa']
        ]
        response = '\n'.join(response_qas)

        icl_inputs.append({
            'query': query,
            'response': response
        })

    messages = [{"role": "system", "content": system_prompt}]
    for sample in icl_inputs:
        messages.append({"role": "user", "content": sample['query']})
        messages.append({"role": "assistant", "content": sample['response']})
    messages.append({"role": "user", "content": user_input})
    # for msg in messages:
    #     print(msg)
    #     print('\n')

    response = client.chat.completions.create(
        model=gpt_model,
        messages=messages,
        max_tokens=1000,
        temperature=1.0,
    )
    motion_scene_qas = response.choices[0].message.content
    # print(response)

    qa_outputs = re.findall(r"Question:(.*?)Answer:(.*?)(?=Question:|$)", motion_scene_qas, re.DOTALL)
    motion_scene_qa_outputs = []
    for qa in qa_outputs:
        question, answer = qa
        motion_scene_qa_outputs.append({
            "question": question.strip(),
            "answer": answer.strip()
        })

    return motion_scene_qa_outputs

def generate_pred_allinfo(
        motion_anns,
        scene_anns,  
        in_context_examples,
        input_template,
        output_template,
        system_prompt,
        client,
        gpt_model,
        **kwargs,
    ):
    object_information = json.dumps(scene_anns['object information'])
    action = motion_anns['action']
    current_key_moments = json.dumps(motion_anns['current key moments'])
    future_key_moments = json.dumps(motion_anns['future key moments'])
    
    user_input = input_template.format(
        object_information, action, current_key_moments, future_key_moments
    )

    icl_inputs = []
    for icl in in_context_examples:
        query = input_template.format(
            json.dumps(icl['object information']),
            icl['action'],
            json.dumps(icl['current key moments']),
            json.dumps(icl['future key moments']),
        )
        response_qas = [
            output_template.format(d['question'], d['answer']) for d in icl['qa']
        ]
        response = '\n'.join(response_qas)

        icl_inputs.append({
            'query': query,
            'response': response
        })

    messages = [{"role": "system", "content": system_prompt}]
    for sample in icl_inputs:
        messages.append({"role": "user", "content": sample['query']})
        messages.append({"role": "assistant", "content": sample['response']})
    messages.append({"role": "user", "content": user_input})
    # for msg in messages:
    #     print(msg)
    #     print('\n')

    response = client.chat.completions.create(
        model=gpt_model,
        messages=messages,
        max_tokens=1000,
        temperature=1.0,
    )
    motion_scene_qas = response.choices[0].message.content
    # print(response)

    qa_outputs = re.findall(r"Question:(.*?)Answer:(.*?)(?=Question:|$)", motion_scene_qas, re.DOTALL)
    motion_scene_qa_outputs = []
    for qa in qa_outputs:
        question, answer = qa
        motion_scene_qa_outputs.append({
            "question": question.strip(),
            "answer": answer.strip()
        })

    return motion_scene_qa_outputs

def generate_dialogue_allinfo(motion_anns,
        scene_anns,  
        in_context_examples,
        input_template,
        output_template,
        system_prompt,
        client,
        gpt_model,
        **kwargs
    ):
    object_information = json.dumps(scene_anns['object information'])
    action = motion_anns['action']
    key_moments = json.dumps(motion_anns['key moments'])
    
    user_input = input_template.format(
        object_information, action, key_moments
    )

    icl_inputs = []
    for icl in in_context_examples:
        query = input_template.format(
            json.dumps(icl['object information']),
            icl['action'],
            json.dumps(icl['key moments'])
        )
        response_qas = [
            output_template.format(d['question'], d['answer']) for d in icl['qa']
        ]
        response = '\n'.join(response_qas)

        icl_inputs.append({
            'query': query,
            'response': response
        })

    messages = [{"role": "system", "content": system_prompt}]
    for sample in icl_inputs:
        messages.append({"role": "user", "content": sample['query']})
        messages.append({"role": "assistant", "content": sample['response']})
    messages.append({"role": "user", "content": user_input})
    # for msg in messages:
    #     print(msg)
    #     print('\n')

    response = client.chat.completions.create(
        model=gpt_model,
        messages=messages,
        max_tokens=1000,
        temperature=0.0,
    )
    motion_scene_qas = response.choices[0].message.content
    # print(response)

    qa_outputs = re.findall(r"Human:(.*?)Agent:(.*?)(?=Human:|$)", motion_scene_qas, re.DOTALL)
    motion_scene_qa_outputs = []
    for qa in qa_outputs:
        question, answer = qa
        motion_scene_qa_outputs.append({
            "question": question.strip(),
            "answer": answer.strip()
        })

    return motion_scene_qa_outputs