def get_math_evaluation_prompt_reference_less(question, model_response):
    return f"""You are an objective evaluator of math word problem solvers. You are given a math problem and a detailed model answer. 
Evaluate the model answer based on how well it solves the math problem.
You should consider multiple aspects, including (but not limited to):
1. Mathematical correctness: Is the final answer correct? 
2. Reasoning: Is the reasoning or calculation process correct and clearly explained?
3. Completeness: Does the answer show all necessary steps or just the final answer?
4. Clarity: Is the answer easy to follow and free of unnecessary information?

Provide a brief textual evaluation explaining your reasoning for the score, highlighting specific strengths and weaknesses.
Then, provide a score from 0 to 1 where:
0 - Completely incorrect or irrelevant answer
0.25 - Incorrect answer with mostly correct reasoning/steps
0.5 - Correct answer with partially correct or incomplete reasoning/steps
0.75 - Fully correct answer with incomplete reasoning/steps
1 - Fully correct answer with clear and complete reasoning

If the final answer is incorrect, score 0.5 or below!
Format your response with the textual evaluation first, followed by the score on a new line like this:
Evaluation score: [score]

Now evaluate the following:
Question: {question}
Model Answer: {model_response}

--- Begin Evaluation ---
Textual Evaluation: [Your textual evaluation here]
Evaluation score: [Your score here]"""


def get_math_evaluation_prompt_reference_based(question, model_response, reference_answer):
    return f"""You are an expert judge evaluating a model-generated answer to a grade school math word problem. You are given:
1. A question
2. A reference (gold) answer, which shows a correct and complete solution
3. A model-generated answer

Your job is to compare the model’s answer to the reference, and assess it holistically.
You should consider multiple aspects, including (but not limited to):

1. Correctness of the final answer
2. Accuracy and validity of reasoning steps
3. Clarity of explanation
4. Completeness of the solution
5. Mathematical coherence and precision

The final score should reflect overall quality, with correctness weighed most heavily, but not exclusively. 
An incorrect final answer with strong reasoning may score moderately; a correct answer with poor reasoning may score lower than perfect.

Provide a brief textual evaluation explaining your reasoning for the score, highlighting specific strengths and weaknesses.
Then, provide a score from 0 to 1 where:
0 - Completely incorrect or irrelevant answer
0.25 - Incorrect answer with mostly correct reasoning/steps
0.5 - Correct answer with partially correct or incomplete reasoning/steps
0.75 - Fully correct answer with incomplete reasoning/steps
1 - Fully correct answer with clear and complete reasoning

Format your response with the textual evaluation first, followed by the score on a new line like this:
Evaluation score: [score]

Now evaluate the following:
Question: {question}
Reference Answer: {reference_answer}
Model Answer: {model_response}

--- Begin Evaluation ---
Textual Evaluation: [Your textual evaluation here]
Evaluation score: [Your score here]
"""

def get_general_evaluation_prompt_reference_less(model_input, model_output, evaluation_criteria_str):

    return f"""You are an impartial judge evaluating the quality of an AI model's response. You will receive:

Input: The text the model was asked to process or respond to.
Output: The model's response text.
Your task is to score the model's response on a scale of 0 to 1, considering the following criteria.
You may also consider other relevant factors that contribute to the overall quality of the response.

Evaluation Criteria:
{evaluation_criteria_str}

Provide a score from 0 to 1 and explain your reasoning clearly and concisely. End the response with 'Evaluation Score: <score>' (e.g., 'Evaluation Score: 0.7').

Input: '{model_input}'
Output: '{model_output}'

--- Begin Evaluation ---
Textual Evaluation: [Your textual evaluation here]
Evaluation score: [Your score here]
"""

def get_general_evaluation_prompt_reference_based(model_input, model_output, reference, evaluation_criteria_str):

    return f"""You are an impartial judge evaluating the quality of an AI model's response. You will receive:

Input: The text the model was asked to process or respond to.
Output: The model's response text.
Reference: The expected reference output.

Your task is to score the model's response on a scale of 0 to 1, considering the following criteria.
You may also consider other relevant factors that contribute to the overall quality of the response.

Evaluation Criteria:
{evaluation_criteria_str}

Provide a score from 0 to 1 and explain your reasoning clearly and concisely. End the response with 'Evaluation Score: <score>' (e.g., 'Evaluation Score: 0.7').

Input: '{model_input}'
Reference: '{reference}'
Output: '{model_output}'

--- Begin Evaluation ---
Textual Evaluation: [Your textual evaluation here]
Evaluation score: [Your score here]
"""

def get_summarization_prompt(evaluation_text: str):
    return \
f"""You are given an evaluation text produced by a judge model. Summarize the text in a few sentences.
Focus on the core reasoning for the score.
Remove redundancies and make it concise while keeping the essential information.
Disregard the score given by the model and focus on the textual feedback.
Use short and simple sentences.

Evaluation Text to Summarize:
{evaluation_text}"""

def get_shortcomings_synthesis_prompt(concatenated_evaluation_texts: str, max_shortcomings: int):
    return \
f"""You are an expert analyst tasked with identifying common themes in evaluation feedback for an AI model's answers. Below is a collection of evaluation texts assessing the quality of different answers.

Your goal is to identify the most significant and frequent types of shortcomings or negative feedback mentioned in these evaluations. Please provide a list of up to {max_shortcomings} concise phrases describing these common issues. Focus on actionable feedback points that could help improve the model's responses.

Guidelines for identifying shortcomings:
1. Look for patterns across multiple evaluations
2. Focus on specific, actionable issues rather than general complaints
3. Consider both content-related issues (accuracy, completeness) and presentation issues (clarity, structure)
4. Prioritize issues that appear frequently or have significant impact
5. Be specific but concise in your descriptions
6. Ensure the issues are distinct and not overlapping. 

Do NOT list positive feedback. Focus only on areas for improvement or reasons for lower scores.
Present the output ONLY as a Python list of strings. Your response MUST start with '[' and end with ']'.

--- Begin Evaluation Texts ---
{concatenated_evaluation_texts}
--- End Evaluation Texts ---

Synthesized List of Common Shortcomings (Python List format ONLY):
"""

def get_shortcomings_clustering_prompt(recurring_issues_list, max_issues):
    return f"""You are given a list of short action items that describe recurring issues found in responses generated by a language model. These items may contain duplicates or very similar entries phrased differently.
Your task is to analyze the list of {len(recurring_issues_list)} issues, remove duplicates and consolidate redundant items into a smaller set of up to {max_issues} distinct, clearly described issues.
Instructions:
- Group nearly identical feedback items that refer to the same concerns.
- If there are two issues assessing different aspects of the same topic - do not merge them.  
- Do not merge issues with the same topic but opposite concerns (e.g: overly verbose / not verbose enough).
- For each group, write a single and clear issue that captures the common idea.
- Ensure that each issue addresses only a single concern or aspect. Do not merge distinct issues with related topics.
- Ensure that the final list avoids redundancy and represents the full variety of distinct concerns from the original list.
- Ensure no important information is lost from the original list — all key concerns must be preserved.
- Reply with a properly formatted Python list of strings (with each element in double quotes) containing the consolidated issues.
Now process the following list:
{recurring_issues_list}
"""


def get_shortcomings_mapping_human_prompt(eval_text, num_shortcomings):
    return f"Evaluation text to analyze:\n```\n{eval_text}\n```\n\nWhich shortcomings (1-{num_shortcomings} from the list provided in the system prompt) are mentioned or implied? Respond ONLY with a Python list of {num_shortcomings} binary values (0 or 1)."


def get_shortcomings_mapping_system_prompt(shortcomings_list):
    shortcomings_list_str = "\n".join([f"{i + 1}. {s}" for i, s in enumerate(shortcomings_list)])
    num_shortcomings = len(shortcomings_list)
    return f"""You are an expert analyst reviewing evaluation feedback.
Your task is to determine which of the following {num_shortcomings} shortcomings (derived from common themes) are mentioned or clearly implied in the specific evaluation text provided.
A shortcoming must be expressed or implied with **negative sentiment** to be marked.

Output a Python list of exactly {num_shortcomings} binary values:
- Use 1 if the shortcoming is clearly **mentioned or implied negatively**
- Use 0 if it is **not mentioned, or is mentioned only positively or neutrally**

You must respond **only** with a Python list of {num_shortcomings} values in the format [0, 1, 0, ...], and each value must be either 0 or 1.
Do not use any other numbers. Do not add text, explanation, or formatting. If unsure, default to 0.

Valid Example: "[1, 0, 1, 0]"
Invalid Examples:
- "[1, 2, 0, 1]"  (2 is invalid)
- "[yes, no, maybe]"  (non-binary)

Shortcoming List:
{shortcomings_list_str}
"""


def get_shortcomings_synthesis_prompt_cont(concatenated_evaluation_texts: str, existing_key_points_texts: str, max_shortcomings: int):
    return \
f"""You are an expert analyst tasked with identifying *new* recurring shortcomings in evaluation feedback for an AI model's answers. Below is a collection of evaluation texts assessing the quality of different answers, along with a list of issues that have already been identified.

Your goal is to discover additional, distinct shortcomings that are not already represented in the existing list. 
Please provide a list of up to {max_shortcomings} concise phrases describing these new common issues. Focus on actionable feedback points that could help improve the model's responses.

Guidelines for identifying new shortcomings:
1. Look for patterns across multiple evaluations
2. Focus on specific, actionable issues rather than general complaints
3. Consider both content-related issues (accuracy, completeness) and presentation issues (clarity, structure)
4. Prioritize issues that appear frequently or have significant impact
5. Be specific but concise in your descriptions
6. Ensure the issues are distinct from those already listed and not overlapping
7. Do NOT repeat any of the existing issues. If a similar issues is already present, exclude the new one. 
Do NOT list positive feedback. Focus only on areas for improvement or reasons for lower scores.


Present the output ONLY as a Python list of strings. Your response MUST start with '[' and end with ']'.

--- Begin Existing Issues ---
{existing_key_points_texts}
--- End Existing Issues ---

--- Begin Evaluation Texts ---
{concatenated_evaluation_texts}
--- End Evaluation Texts ---

Synthesized List of New Shortcomings (Python List format ONLY):
"""

def get_rag_evaluation_prompt_reference_based(question, model_answer, reference):
    return f"""You are an objective evaluator of question answering systems.
You are given a question, a reference answer and a model's response to the question.
Evaluate the model answer based on how well it addresses the question compared to the reference answer.
Consider:
1. Accuracy: Is the information provided factually consistent with the reference?
2. Completeness: Does the model answer cover all key points from the reference?
3. Relevance: Does the model answer directly address the question? Any hallucinations or irrelevant info?
4. Conciseness: Is the answer stated clearly and without unnecessary verbosity?
5. Clarity: Is the answer well-structured, easy to understand, and coherent?

Provide a brief textual evaluation explaining your reasoning for the score, highlighting specific strengths and weaknesses.
Then, provide a score from 0 to 1 where:
0 - Completely irrelevant, incorrect, or harmful answer
0.5 - Partially correct/relevant but significant flaws (missing info, inaccurate, verbose)
1 - Fully correct, complete, relevant, and well-written answer. 
Only flawless answers should get the score 1.

Format your response with the textual evaluation first, followed by the score on a new line like this:
Evaluation score: [score]

Now evaluate this: 
Question: {question}
Reference Answer: {reference}
Model Answer: {model_answer}

--- Begin Evaluation ---
Textual Evaluation: [Your textual evaluation here]
Evaluation score: [Your score here]"""


def get_rag_evaluation_prompt_reference_free(question, documents, model_answer):
    return f"""You are an objective evaluator of question answering systems.
You are given a question, a set of retrieved documents, and a model's response to the question.
Evaluate the model answer based on how well it addresses the question **using only the information from the retrieved documents**.

Consider:
1. **Faithfulness**: Is the information in the model answer factually supported by the retrieved documents, Avoiding hallucinations or fabricated claims?
2. **Completeness**: Does the model answer capture all key points that are present in the documents and relevant to the question?
3. **Relevance**: Does the model answer directly address the question? Does it focus on the pertinent content from the documents?
4. **Conciseness**: Is the answer clearly and succinctly stated, avoiding unnecessary repetition or fluff?
5. **Clarity**: Is the answer well-structured, easy to understand, and coherent?

Provide a brief textual evaluation explaining your reasoning for the score, highlighting specific strengths and weaknesses in how the answer reflects the documents.

Then, provide a score from 0 to 1 where:
0 - The answer is mostly unsupported, irrelevant, or misleading based on the documents.
0.5 - The answer is partially supported but has major omissions, inaccuracies, or irrelevant content.
1 - The answer is fully supported, relevant, complete, and well-written based on the documents.
Only flawless answers should get the score 1. DO NOT score 1 if the answer has any problems in it. 

Format your response with the textual evaluation first, followed by the score on a new line like this:
Evaluation score: [score]

Now evaluate this: 
Question: {question}
Retrieved Documents: {documents}
Model Answer: {model_answer}

--- Begin Evaluation ---
Textual Evaluation: [Your textual evaluation here]
Evaluation score: [Your score here]"""