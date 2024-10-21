from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import tkinter as tk
from tkinter import scrolledtext
from torch.cuda.amp import autocast  # FP16을 사용하기 위해 추가
import threading  # 비동기 처리를 위해 추가

# 토크나이저 및 모델 로드
def prepare_model():
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

    # 패딩 토큰 설정
    tokenizer.pad_token = tokenizer.eos_token  # eos_token을 pad_token으로 사용

    # 모델을 GPU로 이동 (가능할 경우)
    if torch.cuda.is_available():
        model = model.to("cuda")

    return model, tokenizer

# 긴 텍스트 생성을 위한 함수 (여러 번 반복해서 이어붙이는 방식)
def generate_response(model, tokenizer, prompt, max_chunk_length=50, max_iterations=3):
    generated_text = prompt  # 시작 텍스트 설정
    all_text = prompt  # 전체 텍스트를 저장할 변수

    for _ in range(max_iterations):  # 반복 횟수 지정
        # 토큰화 및 attention_mask 추가
        inputs = tokenizer(generated_text, return_tensors="pt", truncation=True).to(model.device)

        # 모델 예측 (FP16 사용)
        with autocast():
            with torch.no_grad():
                outputs = model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=max_chunk_length,  # 새로 생성할 토큰 개수
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False  # 샘플링을 비활성화하여 더 빠른 응답 생성
                )

        # 새로운 텍스트 생성 후 디코딩
        new_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 새로운 텍스트를 전체 텍스트에 이어붙임
        generated_text = new_text[len(generated_text):]  # 중복되는 부분 제외하고 추가
        all_text += generated_text  # 전체 텍스트에 새로운 텍스트 추가

        # 더 이상 생성할 텍스트가 없으면 중단
        if len(generated_text.strip()) == 0:
            break

    return all_text

# GUI 생성 함수
def create_gui():
    # 응답을 출력하는 함수 (비동기 처리)
    def ask_question():
        question = question_entry.get()  # 입력된 질문을 가져옴
        response_text.config(state=tk.NORMAL)
        response_text.delete(1.0, tk.END)  # 이전 응답 삭제
        response_text.insert(tk.END, "Processing...\n")  # 응답 대기 중 메시지 표시
        response_text.config(state=tk.DISABLED)
        
        def get_response():
            response = generate_response(model, tokenizer, question)  # 모델로 응답 생성
            response_text.config(state=tk.NORMAL)  # 응답창 수정 가능하게 설정
            response_text.delete(1.0, tk.END)  # 이전 응답 삭제
            response_text.insert(tk.END, response)  # 새로운 응답 삽입
            response_text.config(state=tk.DISABLED)  # 응답창 수정 불가하게 설정

        # 질문을 비동기로 처리하여 UI 멈추는 현상 방지
        threading.Thread(target=get_response).start()

    # 메인 창
    window = tk.Tk()
    window.title("LLaMA 3.2 Chatbot")

    # 질문 입력 라벨
    question_label = tk.Label(window, text="Enter your question:")
    question_label.pack(pady=10)

    # 질문 입력창
    question_entry = tk.Entry(window, width=50)
    question_entry.pack(pady=10)

    # 질문 제출 버튼
    ask_button = tk.Button(window, text="Ask", command=ask_question)
    ask_button.pack(pady=10)

    # 응답 출력창 (스크롤 가능한 텍스트 박스)
    response_text = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=60, height=15, state=tk.DISABLED)
    response_text.pack(pady=10)

    # GUI 실행
    window.mainloop()

if __name__ == "__main__":
    # 파인 튜닝된 모델을 로드
    model, tokenizer = prepare_model()
    
    # GUI 실행
    create_gui()