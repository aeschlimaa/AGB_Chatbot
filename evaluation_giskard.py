# evaluation
from giskard.rag import KnowledgeBase, generate_testset, evaluate
import pandas as pd
import json
from giskard.rag import QATestset




# # creating a knowledge base
# df = pd.DataFrame([d.page_content for d in documents], columns=["text"])
# df.head(10)
#
# knowledge_base = KnowledgeBase(df)
#
# # creating the Test Set
# testset = generate_testset(
#    knowledge_base,
#    num_questions=60,
#    agent_description="A chatbot answering questions about terms and conditions",
#  )
#
# # save testset to a json file
# testset.save("test-set.json")

# testset is saved as a json file and can be loaded unless pdf has been changed
testset = QATestset.load("test-set.json")

# evaluating model on the test set
answers = []
contexts = []
def answer_fn(question, history=None):
    global answers, contexts
    answer = chain.invoke({"question": question})
    answers.append(answer)
    contexts.append([docs.page_content for docs in retriever.get_relevant_documents(question)])
    return answer

report = evaluate(answer_fn, testset=testset, knowledge_base=knowledge_base)

report.save("test2.html")

# prepare RAGAS: questions and ground_truths
ragas_data = testset.to_pandas()

