DENSE_CAPTION_PROMPT = '''
You are an AI assistant that can understand motion of human in a 3D scene. You will be given the following information of a human motion in a 3D scene. 
Using these information, your job is to generate a detailed caption on the human motion in the 3D scene.
The information contains several fields, and the detailed explanation of each field.
Scene annotations:
{
    "object information": Information of all the objects involved in the motion process of the human. Each object is indexed by a unique id. The information of each object is organized in the form of {<object id>: {"category": the general category name of the object, "referral": a list of references describing the object}}.
}
Motion annotations:
{
    "action": an overall description of the activity that the person is engaging with during the whole motion sequence,
    "key moments": a list of multi-perspective descriptions on several key moments evenly sampled from the whole motion, in time order. Descriptions of each key moment includes following aspects:
    {
        "human-scene contacts": a list that specifies all the human body parts that are in contact with the objects in scene at this moment. Each item in the list consists of {<body part name>, <object id>}.
        "human-scene spatial relations": a list that specifies all the spatial relations between the person and the objects in scene at this moment. Each item in the list consists of {<distance>, <orientation>, <object id>}.
    }
}
Your caption should refer to information of both the human and scene. Describe from the following aspects: the human's activity, the human's relative position to other objects, the human's contacts with other objects. Describe as if you are directly perceiving a human motion. The length of caption should be around 50 words. You will be given some examples in the following conversations. Follow the style and length of these examples to generate your output.
'''

QA_ACTION_PROMPT = '''
You are an AI assistant that can understand motion of human in a 3D scene. You will be given the following information of a human's activity in a 3D scene.
The information is listed as follows:
{{  
    "action": description of the activity that the person is engaging with during the whole motion sequence.
}}

{}

Formulate question and answer as if you are directly perceiving a natural scene and human motion, which means that you should not mention that your answer is coming from the scene and motion annotations!
'''

QA_ALLINFO_PROMPT = '''
You are an AI assistant that can understand motion of human in a 3D scene. You will be given the following information of a human motion in a 3D scene.
The information contains several fields, and is listed as follows:
Scene annotations:
{{
    "object information": Information of all the objects involved in the motion process of the human. Each object is indexed by a unique id. The information of each object is organized in the form of [<object id>: ["category": the general category name of the object, "referral": a list of references describing the object]].
}}
Motion annotations:
{{
    "action": an overall description of the activity that the person is engaging with during the whole motion sequence,
    "key moments": a list of multi-perspective descriptions on several key moments evenly sampled from the whole motion, in time order. Descriptions of each key moment includes following aspects:
    {{
        "pose": a detailed description on the part-level body pose of a person at this moment.
        "human-scene contacts": a list that specifies all the human body parts that are in contact with the objects in scene at this moment. Each item in the list consists of [<body part name>, <object id>].
        "human-scene spatial relations": a list that specifies all the spatial relations between the person and the objects in scene at this moment. Each item in the list consists of [<distance>, <orientation>, <object id>].
    }}
}}

{}

Formulate question and answer as if you are directly perceiving a natural scene and human motion, which means that you should not mention that your answer is coming from the scene and motion annotations, such as index of objects, `key moments', or `spatial relations'! Try to use diverse sentence patterns in the question.
'''

PRED_PROMPT = '''
You are an AI assistant that can understand motion of human in a 3D scene. You will be given the following information of a human motion in a 3D scene, which is split into current motion and future motion. The information contains several fields, and is listed as follows:
Scene annotations:
{{
    "object information": Information of all the objects involved in the motion process of the human. Each object is indexed by a unique id. The information of each object is organized in the form of [<object id>: ["category": the general category name of the object, "referral": a list of references indicating the object's relations with other objects]].
}}
Motion annotations:
{{
    "action": an overall description of the activity that the person is engaging with during the whole motion sequence,
    "current key moments": a list of multi-perspective descriptions on several key moments evenly sampled from the current motion, in time order. Descriptions of each key moment includes following aspects:
    {{
        "pose": a detailed description on the part-level body pose of a person at this moment. Note that the description might be over meticulous and you are suggested to only choose the parts that are most relevant to the overall human movements and activities.
        "human-scene contacts": a list that specifies all the human body parts that are in contact with the objects in scene at this moment. Each item in the list consists of [<body part name>, <object id>].
        "human-scene spatial relations": a list that specifies all the spatial relations between the person and the objects in scene at this moment. Each item in the list consists of [<distance>, <orientation>, <object id>].
    }},
    "future key moments": a list of multi-perspective descriptions on several key moments evenly sampled from the future human motion, in time order. Descriptions of each key moment include the same aspects as in the 'current key moments' field.
}}

{}

Formulate question and answer as if you are directly perceiving a natural scene and human motion, which means that you should not mention that your answer is coming from the scene and motion annotations, such as index of objects, `key moments', or `spatial relations'!
'''

### task-specific templates
ACTIVITY = '''
Your job is to use all these information of the human and the scene to generate some question-answer pairs asking about the human's activity. Try to use diverse question styles, but only ask directly about activity, and do not ask about other aspects.
'''

HOI = '''
Your job is to use all these information of the human and the scene to generate some question-answer pairs asking about knowledge related to human-object interactions. Such as: the type of interaction, the body part of the human that is in contact with an object, the name of object that the human is in contact with. Do not ask about aspects other than human-object interactions.
'''

HOI_INTERACTION = '''
Your job is to use all these information of the human and the scene to generate a question-answer pair asking about the type of interaction between the human and a certain object in the scene. Do not ask about other aspects.
'''

HOI_OBJECT = '''
Your job is to use all these information of the human and the scene to generate 1-3 question-answer pairs asking about the object the human (or a specified body part) is in contact with. The question should be asking about the object, but not the body part. Also, do not ask about other aspects.
'''

HOI_PART = '''
Your job is to use all these information of the human and the scene to generate 1-3 question-answer pairs asking about the body part of the human that is in contact with an object. The question should be asking about the body part, but not the object. Also, do not ask about other aspects.
'''

LOC = '''
Your job is to use all these information of the human and the scene to generate some question-answer pairs asking about knowledge related to position of humans. Such as: the position of the human, the orientation of a certain object in relation to the human, the object that is at a certain orientation of the human. Do not ask about aspects other than positions and orientations.
'''

LOC_OBJECT = '''
Your job is to use all these information of the human and the scene to generate 1-3 question-answer pairs asking about the object that is at a certain orientation of the human. Be aware that since the human is moving, there might be multiple objects at the certain orientation during his motion. Do not ask about other aspects.
'''

LOC_ORIENT = '''
Your job is to use all these information of the human and the scene to generate 1-3 question-answer pairs asking about the orientation of a certain object in relation to the human. Answer brief. Do not ask about other aspects.
'''

LOC_POSITION = '''
Your job is to use all these information of the human and the scene to generate a question-answer pair asking about the position of the human. Only answer the positions of human and objects. Do not ask about other aspects.
'''

PRED = '''
Your job is to use all these information to generate a question-answer pair asking about the future activities of the human. Such as: future movements, future location, future intents. Do not include too detailed information in your answer, such as details of pose, contact, and body parts. In this setting, assume that the answerer can only observe the 'current key moments' part of the human motion, and give the answer based on the 'future key moments' part of the human motion.
'''

PRED_INTENT = '''
Your job is to use all these information to generate a question-answer pair asking about the future intent of the human. In this setting, assume that the answerer can only observe the person's "current key moments", and give the answer based on the "future key moments". Do not ask about other aspects.
'''

PRED_MOVEMENT = '''
Your job is to use all these information to generate a question-answer pair asking about the future movements of the human. In this setting, assume that the answerer can only observe the 'current key moments' part of the human motion, and give the answer based on the 'future key moments' part of the human motion. Focus on the location, and do not answer information other than movements, such as action or pose.
'''

PLANNING = '''
Your job is to imagine yourself as an intelligent agent, use all these information as observations of a human activity, and formulate question-answer pairs asking about what you can do to help the human. You can ask about high-level plannings, which generally state a task you could perform to help the person. Or you can ask about low-level plannings, which give a step-wise decomposition on the action steps to perform this plan.
You should try to make the task planning relevant to the human's position, orientation and pose status, and avoid generating plans that can be conducted without knowing the status of the human in the scene.
'''

DIALOGUE = '''
Your job is to use all these information to generate a meaningful 2-4 round conversation about the human motion in the scene. The conversation is between the human which is presented in the scene, and an intelligent agent which has the full knowledge of the scene. The dialogue should be the human inquiring for certain assistance related to his/her current activities, and the agent providing helpful and accurate answers. You should try to make most of the dialogue relevant to the human's position, orientation and pose status, and avoid generating questions that could be answered without perceiving the human's current status in the scene. Do not contain unnecessary conversations like "Thank you", "You are welcome".
'''

OPEN = '''
Your job is to use all these information of the human and the scene to generate 3-5 question-answer pairs asking about the human in the scene. Your question can be diverse, focusing on knowledge such as human activities, movements, positions, interactions between human and scene, and so on. Your should include 1-2 questions that involves complex reasoning. Try to be imaginative at the question types, and use language styles that are as various as possible.
'''