from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_correctness,
    context_precision,
    context_recall,
)

questions = [
    "Wie werden Münzen oder andere Wertsachen bei BEKB verwahrt?"
]

ground_truths = [[
    "Bei der BEKB werden Münzen oder andere Wertsachen in einem verschlossenen Depot verwahrt."
]]

answers = []
contexts = []


for query in questions:
  docs = retriever.get_relevant_documents(query)
  answers.append(chain.invoke({"question":query}))
  contexts.append([docs.page_content for docs in retriever.get_relevant_documents(query)])

# Example data
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "reference": ground_truths
}

# Convert the data to a Hugging Face Dataset
dataset = Dataset.from_dict(data)

# Define the metrics you want to evaluate
metrics = [
    faithfulness,
    answer_relevancy,
    answer_correctness,
    context_precision,
    context_recall,
]

# Evaluate the dataset using the selected metrics
results = evaluate(dataset, metrics)

# Display the results
for metric_name, score in results.items():
    print(f"{metric_name}: {score:.2f}")

