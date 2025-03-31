# evaluation
from giskard.rag import KnowledgeBase, generate_testset, evaluate
import pandas as pd
import json
from giskard.rag import QATestset

from agb_chatbot import pages



# creating a knowledge base
df = pd.DataFrame([d.page_content for d in pages], columns=["text"])
df.head(10)

knowledge_base = KnowledgeBase(df)

# creating the Test Set
testset = generate_testset(
    knowledge_base,
    num_questions=60,
    agent_description="A chatbot answering questions about terms and conditions",
   )

# defining answer_fn, that retrieves documents (contexts) and answer the questions
answers = []
contexts = []
def answer_fn(question, history=None):
    global answers, contexts
    answer = chain.invoke({"question": question})
    answers.append(answer)
    contexts.append([docs.page_content for docs in retriever.get_relevant_documents(question)])
    return answer

# the actual evaluation
report = evaluate(answer_fn, testset=testset, knowledge_base=knowledge_base)

# store the report
report.save("results_giskard")

# preparation RAGAS: questions and ground_truths
ragas_data = testset.to_pandas()
