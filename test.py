import json
from predict.predict import main


if __name__ == '__main__':
    with open("../data/data.json", 'r') as fr:
        result = json.load(fr, encoding="utf-8")['data']
    count = 0
    for x in result:
        x = x["paragraphs"]
        count += len(x[0]['qas'])
    print(count)
    # for r in result[:50]:
    #     print(r['id'],r['answer'])
        # print(r)

    test_data_path = "../data/data.json"
    data_dir = "../result/"
    # main(test_data_path, data_dir)
    # result = []
    # with open("../result/nbest_predictions.json", "r", encoding="utf-8") as fr:
    #     data = json.load(fr)
    # print(len(data))
    # for key, value in data.items():
    #     answer = value[0]["text"].replace(' ', '')
    #     res = {"answer": answer, "id": key}
    #     result.append(res)
    #
    # with open("../result/result.json", 'w') as fr:
    #     json.dump(result, fr, ensure_ascii=False)
