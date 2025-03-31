from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_correctness,
    context_precision,
    context_recall,
)

# preparing the data 
questions = [ragas_data["question"].tolist()][0]
ground_truths = [ragas_data["reference_answer"].tolist()][0]
answers = answers
contexts = contexts

# create dataset
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
results

# calculate the average for generation/retriever
(sum(results["faithfulness"])/60 + sum(results['answer_relevancy'])/60)/2
(sum(results['context_precision'])+sum(results['context_recall']))/120


