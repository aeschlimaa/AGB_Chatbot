# evaluation
from giskard.rag import KnowledgeBase, generate_testset, evaluate
import pandas as pd

# creating a knowledge base
df = pd.DataFrame([d.page_content for d in documents], columns=["text"])
df.head(10)

knowledge_base = KnowledgeBase(df)

# creating the Test Set
testset = generate_testset(
    knowledge_base,
    num_questions=60,
    agent_description="A chatbot answering questions about terms and conditions",
)

# save test set to a jsonl file
testset.save("test-set.jsonl")

# evaluating model on the test set
def answer_fn(question, history=None):
    return chain.invoke({"question": question})

report = evaluate(answer_fn, testset=testset, knowledge_base=knowledge_base)

display(report)




# refinements

# compare different models
"https://github.com/svpino/llm/blob/main/evaluation/notebook.ipynb"