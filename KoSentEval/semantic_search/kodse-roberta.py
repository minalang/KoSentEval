import numpy as np
import torch
import os
import random
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
# random.seed(1111)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'

def pytorch_cos_sim(a, b):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    This function can be used as a faster replacement for 1-scipy.spatial.distance.cdist(a,b)
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.Tensor(a) #Tensor
        a = a.to(device)
        
    if not isinstance(b, torch.Tensor):
        b = torch.Tensor(b)
        b = b.to(device)
        
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
        a = a.to(device)
        
    if len(b.shape) == 1:
        b = b.unsqueeze(0)
        b = b.to(device)
        
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    return torch.matmul(a_norm, b_norm.transpose(0, 1))


# 임베딩값 얻기
def get_embeddings(text):
    # 모델 설정
    model = AutoModel.from_pretrained('minalang/KoDSE-roberta').to(device)  # or 'BM-K/KoSimCSE-roberta-multitask'
    tokenizer = AutoTokenizer.from_pretrained('minalang/KoDSE-roberta')  # or 'BM-K/KoSimCSE-roberta-multitask'
    # 텍스트 처리
    inputs_output = tokenizer(text, max_length=64, padding='max_length', truncation=True, return_tensors='pt').to(device)
    embeddings_output, _ = model(**inputs_output, return_dict = False)
    average_embedding = torch.mean(embeddings_output, dim=1)
    result_embedding = average_embedding.tolist()[0]
    return result_embedding

def main():
    result_csv = pd.DataFrame(index = ['top1acc', 'top3acc', 'top5acc'])   
    data = pd.read_csv('./semantic_search_5000.csv')
    corpus = data['answer'].tolist()
    corpus_embeddings = []
    for c in tqdm(corpus):
        embedding = get_embeddings(c)
        corpus_embeddings.append(embedding)
    corpus_embeddings2 = np.array(corpus_embeddings)
    
    
    f = open('./result/kodse-roberta.txt', 'w')
    
    # calculate accuracy
    top_1_acc = 0
    top_3_acc = 0
    top_5_acc = 0
    
    # Query sentences:
    queries = data['question'].tolist()

    # Find the closest 1, 3, 5 sentences of the corpus for each query sentence based on cosine similarity
    top_3= 3
    top_5 = 5
    for i in tqdm(range(len(queries))): #query in queries:
        # 질문 query embedding추출
        query_embedding = get_embeddings(queries[i])
        ## 정답 후보가 되는 embedding목록 추출

        # 제외할 특정 번호
        excluded_number = i
        excluded_number_list = [excluded_number]
        
        # 제외할 번호를 제외하고 나머지 인덱스 추출
        remaining_indices = [index for index in range(len(corpus_embeddings)) if index != excluded_number]

        # 랜덤하게 1000개의 변수 선택
        random_indices = random.sample(remaining_indices, 99)
        random_result = random_indices+excluded_number_list
        
        # 선택된 변수의 인덱스와 값을 가지고 있는 새로운 리스트 생성
        result_list = [(index, corpus_embeddings[index]) for index in random_result]

        candidate_embeddings = [tpl[1] for tpl in result_list]
                
        cos_scores = pytorch_cos_sim(query_embedding, candidate_embeddings)
        cos_scores = cos_scores.cpu().detach().numpy()

        top_5_results = np.argpartition(-cos_scores, range(5))[0][:5]
        
        sim_list = [result_list[index][0] for index in top_5_results]
        
        top_1_result = sim_list[0]
        top_3_results = sim_list[0:top_3]

        if top_1_result == i:
            top_1_acc+=1
        if i in top_3_results:
            top_3_acc+=1
        if i in sim_list:
            top_5_acc+=1
        # 파일에 쓰기
        # query
        q = f'{i}번째 query: {queries[i]}\n'
        f.write(q)
        
        # 유사도가 높은 문장과 그 유사도
        j = 0
        for idx in sim_list:
            d = f"idx {idx}| {corpus[idx].strip()} (Score: {cos_scores[0][j]:.4f})\n"
            #d = f'{corpus[idx]}, "(Score: {top_10_results[0][idx]})\n'
            f.write(d)
            #print(corpus[idx].strip(), "(Score: %.4f)" % (cos_scores[0][idx])) #.strip()
            j+=1
        f.write('\n')
    # 각 파일의 semantic search 정확도 파일에 저장
    query_num = len(queries)
    acc = [top_1_acc/query_num, top_3_acc/query_num, top_5_acc/query_num]
    result_csv['result'] = acc
    print(result_csv)
    f.close()
    result_csv.to_csv('./result/kodse-roberta.csv', encoding = 'utf-8-sig')

if __name__ == '__main__':
    main()