import pandas as pd
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import sent_tokenize

type = "obf" # org / obf

file_path = f"{type}-chat.csv"
data = pd.read_csv(file_path)

# Filter out only the columns with "answer w/o code" in their name
answer_wo_code_columns = [col for col in data.columns if "answer" in col and "w/o code" in col]
filtered_data = data[answer_wo_code_columns]
results = []

for index, row in filtered_data.iterrows():
    arr = []
    for column in filtered_data.columns:
        if not pd.isna(row[column]):
            arr.append(str(row[column]))

    # 텍스트 토큰화
    # nltk.download('punkt_tab')
    tokenized_data = []
    for ans in arr:
        for ph in ans:
            tokenized_data.append(sent_tokenize(ph))

    smoothie = SmoothingFunction().method4

    # Self-BLEU 점수 계산
    self_bleu_scores = []
    for i, hypothesis in enumerate(tokenized_data):
        # 현재 문장을 가설로, 나머지 문장을 참조로 사용
        references = tokenized_data[:i] + tokenized_data[i + 1:]  # 현재 문장을 제외한 모든 문장
        if references:
            bleu_score = sentence_bleu(references, hypothesis, smoothing_function=smoothie)
            self_bleu_scores.append(bleu_score)

    # Self-BLEU 평균 계산
    average_self_bleu = np.mean(self_bleu_scores) if self_bleu_scores else 0

    # 결과 출력
    print(data.iloc[index, 0], average_self_bleu)
    results.append({"Index_Value": data.iloc[index, 0], "Average_Self_BLEU": average_self_bleu})

output_df = pd.DataFrame(results)
output_df.to_csv(f"{type}-self-bleu.csv", index=False)
