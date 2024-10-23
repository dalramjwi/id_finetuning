import os
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# 1. JSON 파일 경로 설정
data_file_path = os.path.join(os.path.dirname(__file__), 'data/story_form.json')

# 2. JSON을 토대로 데이터셋 로드
dataset = load_dataset('json', data_files=data_file_path)

# 3. 로그 설정 
log_file_handler = logging.FileHandler('training.log', encoding='utf-8')
log_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(log_file_handler)

# 4. 폴더 내에서 가장 높은 버전의 모델을 찾는 함수
def get_latest_model_version(folder):
    existing_versions = [d for d in os.listdir(folder) if d.startswith("fine_tuned_model_v")]
    if not existing_versions:
        return None
    latest_version = max([int(d.split('_v')[-1]) for d in existing_versions])
    return latest_version

# 5. 모델 저장 경로 설정
model_folder = "./fine_tuned_models"
os.makedirs(model_folder, exist_ok=True)

# 6. 가장 최근 버전 모델 불러오기 or 새로 생성
latest_version = get_latest_model_version(model_folder)
if latest_version is not None:
    latest_model_path = f"{model_folder}/fine_tuned_model_v{latest_version}"
    logging.info(f"기존 학습된 모델 {latest_model_path} 로드")
    tokenizer = AutoTokenizer.from_pretrained(latest_model_path)
    model = AutoModelForCausalLM.from_pretrained(latest_model_path)
    new_version = latest_version + 1
else:
    logging.info(f"새 모델 로드")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    new_version = 1

# 7. 패딩 토큰을 EOS 토큰으로 설정
tokenizer.pad_token = tokenizer.eos_token
logging.info("패딩 토큰을 EOS 토큰으로 설정")

# 8. 데이터 전처리 함수
def preprocess_function(examples):
    inputs = tokenizer(examples['input'], padding="max_length", truncation=True, max_length=50)
    outputs = tokenizer(examples['output'], padding="max_length", truncation=True, max_length=50)
    inputs["labels"] = outputs["input_ids"]
    return inputs

logging.info("데이터 전처리 함수 정의 완료")

# 9. 데이터셋에 전처리 적용
tokenized_datasets = dataset.map(preprocess_function, batched=True)
logging.info("데이터셋에 전처리 적용 완료")

# 10. 학습 설정
training_args = TrainingArguments(
    output_dir="./results",            
    per_device_train_batch_size=1,     
    num_train_epochs=10,               
    logging_dir="./logs",              
    logging_steps=50,                  
    save_steps=500,                    
    eval_strategy="no",                
    save_total_limit=1,                
    fp16=False,                        
    gradient_accumulation_steps=8,     
)
logging.info("학습 설정 완료")

# 11. 트레이너 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],  # 모든 데이터셋을 학습에 사용
    eval_dataset=None  # 검증 데이터셋은 사용하지 않음
)
logging.info("트레이너 설정 완료")

# 12. 모델 파인튜닝 반복 실행
if latest_version is not None:
    logging.info(f"{latest_model_path}에 파인튜닝을 이어서 수행합니다.")
else:
    logging.info(f"새 모델을 학습합니다.")

for i in range(50):
    logging.info(f"===== {i+1}번째 학습 시작 =====")
    trainer.train()
    logging.info(f"===== {i+1}번째 학습 완료 =====")

# 13. 학습된 모델 저장 (새 버전으로 저장)
new_model_path = f"{model_folder}/fine_tuned_model_v{new_version}"
model.save_pretrained(new_model_path)
tokenizer.save_pretrained(new_model_path)
logging.info(f"학습된 모델 {new_model_path} 저장 완료")
