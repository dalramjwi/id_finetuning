from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import os
import torch

# JSON 데이터 파일 경로 설정
data_file_path = os.path.join(os.path.dirname(__file__), 'data/story_form.json')

# 데이터셋 로드
dataset = load_dataset('json', data_files=data_file_path)

# 모델과 토크나이저 로드
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 패딩 토큰을 EOS 토큰으로 설정
tokenizer.pad_token = tokenizer.eos_token

# 데이터 전처리 함수
def preprocess_function(examples):
    inputs = tokenizer(examples['input'], padding="max_length", truncation=True, max_length=50)
    outputs = tokenizer(examples['output'], padding="max_length", truncation=True, max_length=50)
    inputs["labels"] = outputs["input_ids"]
    return inputs

# 데이터셋에 전처리 적용
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# GPU 사용 여부 확인
if torch.cuda.is_available():
    device = torch.device('cuda')
    model = model.to(device)
    print(f"GPU 사용 중: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("GPU를 사용할 수 없습니다. CPU를 사용 중입니다.")

# 학습 설정 (로그 추가 및 배치 크기 감소)
training_args = TrainingArguments(
    output_dir="./results",         # 결과 저장 경로
    per_device_train_batch_size=1,  # 배치 크기를 1로 줄임
    num_train_epochs=10,            # 학습 에폭 수 증가
    logging_dir="./logs",           # 로그 저장 경로
    logging_steps=1,                # 매 스텝마다 로그 출력
    save_steps=1000,                # 모델 저장 주기
    eval_strategy="steps",          # 평가 전략 (최신 버전 적용)
    save_total_limit=2,             # 저장할 체크포인트 수 제한
    load_best_model_at_end=True,    # 최적 모델 로드
    evaluation_strategy="steps"     # 학습 중간에 평가
)

# 트레이너 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train']
)

# 모델 파인튜닝 시작
trainer.train()

# 학습된 모델 저장
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# 모델 테스트
def test_model():
    print("===== 테스트 시작 =====")
    inputs = tokenizer("긍정, 부정, 긍정", return_tensors="pt").input_ids
    if torch.cuda.is_available():
        inputs = inputs.to('cuda')  # GPU 사용 시 입력도 CUDA로 이동
    
    outputs = model.generate(inputs, max_new_tokens=50)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("모델 응답:", result)
    print("===== 테스트 끝 =====")

# 데이터셋 확인 로그
print("===== 데이터셋 확인 =====")
print(tokenized_datasets['train'])

# 학습이 완료된 후 테스트 실행
if __name__ == "__main__":
    print("학습이 완료되었습니다.")
    test_model()
