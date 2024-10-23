import os
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch

# 1. 로그 설정 
log_file_handler = logging.FileHandler('test.log', encoding='utf-8')
log_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(log_file_handler)

# 2. 폴더 내에서 가장 높은 버전의 모델을 찾는 함수
def get_latest_model_version(folder):
    existing_versions = [d for d in os.listdir(folder) if d.startswith("fine_tuned_model_v")]
    if not existing_versions:
        return None
    latest_version = max([int(d.split('_v')[-1]) for d in existing_versions])
    return latest_version

# 3. 모델 저장 경로 설정
model_folder = "./fine_tuned_models"
latest_version = get_latest_model_version(model_folder)

if latest_version is not None:
    latest_model_path = f"{model_folder}/fine_tuned_model_v{latest_version}"
    logging.info(f"최신 학습된 모델 {latest_model_path} 로드")
    tokenizer = AutoTokenizer.from_pretrained(latest_model_path)
    model = AutoModelForCausalLM.from_pretrained(latest_model_path)
else:
    logging.error("학습된 모델이 존재하지 않습니다.")
    raise ValueError("학습된 모델이 없습니다. 먼저 모델을 학습시켜주세요.")

# 4. JSON 파일 경로 설정 및 데이터셋 로드
data_file_path = os.path.join(os.path.dirname(__file__), 'data/story_form.json')
dataset = load_dataset('json', data_files=data_file_path)

# 5. 모델 테스트 함수
def test_model(model, tokenizer, dataset):
    logging.info("===== 모델 테스트 시작 =====")
    
    for case in dataset['train']:
        input_text = case["input"]
        expected_output = case["output"]

        # 입력을 토큰화하고 attention_mask 추가
        inputs = tokenizer(input_text, return_tensors="pt", padding=True)
        attention_mask = inputs['attention_mask']
        
        # CPU 사용 설정
        inputs = inputs.to('cpu')
        attention_mask = attention_mask.to('cpu')
        
        # 모델을 통해 텍스트 생성 (attention_mask 사용)
        outputs = model.generate(
            inputs['input_ids'], 
            attention_mask=attention_mask, 
            max_new_tokens=20,  
            pad_token_id=tokenizer.eos_token_id  
        )
        
        # 출력 디코딩
        model_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 결과 출력
        logging.info(f"입력: {input_text}")
        logging.info(f"모델의 출력: {model_output}")
        logging.info(f"기대되는 출력: {expected_output}")
        logging.info("="*50)
    
    logging.info("===== 모델 테스트 완료 =====")

# 6. 테스트 실행
test_model(model, tokenizer, dataset)
