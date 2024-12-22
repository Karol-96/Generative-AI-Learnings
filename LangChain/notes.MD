1) Mitigating LLM Limitations
Retrieval augmentation: This technique accesses knowledge bases to supplement an
LLM’s outdated training data, providing external context and reducing hallucination risk.
• Chaining: This technique integrates actions like searches and calculations.
• Prompt engineering: This involves the careful crafting of prompts by providing critical
context that guides appropriate responses.
• Monitoring, filtering, and reviews: This involves ongoing and effective oversight of
emerging issues regarding the application’s input and output to detect issues. Both manual reviews and automated filters then correct potential problems with the output. This
includes the following:
a. Filters, like block lists, sensitivity classifiers, and banned word filters, can automatically flag issues.
b. Constitutional principles monitor and filter unethical or inappropriate content.
c. Human reviews provide insight into model behavior and output.
• Memory: Retains conversation context by persisting conversation data and context across
interactions.
• Fine-tuning: Training and tuning the LLM on more appropriate data for the application
domain and principles. This adapts the model’s behavior for its specific purpose.


2) Elicit Prompting
Elicit prompting refers to crafting prompts in a way that encourages a model or individual to provide more detailed or specific responses. The aim is to extract high-quality, actionable information by carefully guiding the context or the way a question is posed.

In the context of AI, elicit prompting might involve:

Specifying the task clearly: Ensuring the AI understands exactly what is being asked.

Example: Instead of asking, "What is the weather?", you could say, "What is the current weather forecast for Kathmandu?"
Using examples: Providing examples in the prompt to clarify the expected format or type of response.

Example: "Rewrite this sentence with better grammar: 'I goes to school every day.'"
Adding constraints or focus areas: Indicating specific areas to focus on or boundaries to stay within.

Example: "Explain Newton's laws of motion in simple terms suitable for a 10-year-old."
Elicit prompting is essential for improving the quality and relevance of responses, especially in open-ended or complex tasks.



3) Chain-of-Thought (CoT) Reasoning
Chain-of-thought reasoning is a technique where an AI or individual explicitly outlines the steps of reasoning or decision-making to arrive at a conclusion. In the context of AI, CoT prompting encourages the model to break down its thought process into logical, step-by-step explanations.

Why It’s Useful:
Improves Problem Solving: Helps tackle complex or multi-step tasks like math problems, logical reasoning, or coding challenges.
Enhances Transparency: Makes the reasoning process clear and easier to debug or validate.
Boosts Accuracy: Breaking problems into smaller steps reduces errors and ensures the solution covers all aspects.
Example:
Without CoT reasoning:

Q: If there are 10 apples and you give away 3, how many do you have left?
A: 7.
With CoT reasoning:

Q: If there are 10 apples and you give away 3, how many do you have left?
A: First, start with the total number of apples, which is 10. Next, subtract the number of apples given away (3). The remaining number of apples is 10 - 3 = 7. So, the answer is 7.
How to Prompt for CoT Reasoning:
Add instructions like:
"Think step by step."
"Explain your reasoning."
"List out the steps involved in solving this problem.


