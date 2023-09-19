import gradio as gr
from transformers import pipeline

def generate_response(prompt):

    system_message ="""Below is an instruction that describes a task.Write a response that appropriately completes the request.Please wrap your code answer using ``` """

    # system_message = """You are a helpful, respectful and honest assistant.Your job is to generate python code to solve the following coding problem that obeys the constraints and you also have to give some test cases as an example and show the output.
    # Explain the code after the code completion.Ask the user for any another queries.Please wrap your code answer using ```"""

    prompt_template= f'''
    [INST]
    <<sys>>
    {system_message}
    <</sys>>
    {prompt}
    [/INST]
    '''
    # Generate a response using the pipeline
    pipe = pipeline(
        "text-generation",
        model=model, # Luffy/codellama-2-7b-Instruct-hf-Fine-tuned
        tokenizer=tokenizer, # AutoTokenizer.from_pretrained(model_name)
        max_length=1024,
        temperature=0.3,
        top_p=0.95,
        repetition_penalty=1.15
    )

    generated_text = pipe(prompt_template)[0]['generated_text']
    # Extract content between triple backticks
    code_start = generated_text.find("```")
    code_end = generated_text.rfind("```")
    if code_start != -1 and code_end != -1:
        generated_text = generated_text[code_start + 3:code_end].strip()

    # Remove any remaining unwanted text
    generated_text = generated_text.replace("<</sys>>", "").replace("[/INST]", "").strip()
    return generated_text
title = "CodeLlama-13B for Code Generation "
examples = [
    'Write a python code to find the Fibonacci series.',
    'Write a python code for Merge Sort.',
    'Write a python code for Binary search.',
    'Write a python code for the Longest subsequence.'
]

gr.Interface(
    fn=generate_response,
    inputs=gr.inputs.Textbox(label="Enter your prompt here..."),
    outputs=gr.outputs.Textbox(),
    title=title,
    examples=examples
).launch()