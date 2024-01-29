import requests
from time import sleep, time
import json
from pathlib import Path
from copy import deepcopy
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from statistics import mean

def dummy_task(data, poll_interval=0.05, max_attempts=10):
    base_uri = r'http://localhost:8086'
    predict_task_uri = base_uri + '/churn/predict'
    task = requests.post(predict_task_uri, json=data)
    task_id = task.json()['task_id']
    predict_result_uri = base_uri + '/churn/result/' + task_id
    attempts = 0
    result = None
    while attempts < max_attempts:
        attempts += 1
        result_response = requests.get(predict_result_uri)
        if result_response.status_code == 200:
            result = result_response.json()['probability']
            break
        # print(result_response.json())
        sleep(poll_interval)
    return result


def dummy_classification_task(data, poll_interval=0.05, max_attempts=10):
    base_uri = r'http://localhost:8086'
    predict_task_uri = base_uri + '/news/predict'
    task = requests.post(predict_task_uri, json={"content": data})
    task_id = task.json()['task_id']
    predict_result_uri = base_uri + '/news/result/' + task_id
    attempts = 0
    result = None
    while attempts < max_attempts:
        attempts += 1
        result_response = requests.get(predict_result_uri)
        if result_response.status_code == 200:
            result = result_response.json()['probability']
            break
        # print(result_response.json())
        sleep(poll_interval)
    return result


if __name__ == '__main__':
    
    # with open(Path(__file__).parent / 'data/sample.json') as f:
    #     test_body = json.load(f)[0]

    #     num = 1000
    #     batches = []
    #     for i in range(num):
    #         body = deepcopy(test_body)
    #         body["Total_Trans_Amt"] = i + 1
    #         batches.append(body)

    with open(Path(__file__).parent / 'data/news_sample.json') as f:
        batches = json.load(f)

    print(f"Total batches: {len(batches)}")

    results = []
    t_0 = time()

    poll_interval = 0.25
    max_attempts = 10 // poll_interval
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_input = {
            executor.submit(dummy_classification_task, batch, poll_interval=poll_interval, max_attempts=max_attempts): 
            batch 
            for batch in batches
        }
        cnt = 0
        for future in concurrent.futures.as_completed(future_to_input):
            
            batch = future_to_input[future]
            try:
                result = future.result()
            except Exception as exc:
                print('%r generated an exception (type: (%s)): %s' % (batch, type(exc), exc))
            else:
                results.append(result)
            
            cnt += 1
            if cnt % int(0.1 * len(batches)) == 0:
                print(f"Progress: {cnt / len(batches) * 100: .2f}% within {time() - t_0: .1f}s")

    t_final = time() - t_0
    assert len(results) == len(batches), f"Only got {len(results)} results, expected {len(batches)}"
    assert all([r is not None for r in results]), f"Got None in results: {results}"

    print(f"Average inference time: {t_final / len(batches): .3f}s")
    print(f"Average prediction: {mean(results): .3f}")