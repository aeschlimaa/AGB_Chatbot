import time

retriever_speed = []
for question in questions:
    start_time = time.time()
    retriever.get_relevant_documents(question)
    end_time = time.time()
    retriever_speed.append(end_time - start_time)
sum(retriever_speed) / len(retriever_speed)

generation_speed = []
for question in questions:
    print("start")
    start_time = time.time()
    chain.invoke({"question": question})
    end_time = time.time()
    generation_speed.append(end_time - start_time)
    print("end")
sum(generation_speed) / len(generation_speed)
