import subprocess
import re

# 파일에서 데이터를 읽고 정렬하는 함수
def read_and_sort_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        data = []
        for line in lines:
            parts = line.strip().split(',')
            last_col_value = float(parts[-1])  # 마지막 열의 값을 float으로 변환
            input_string = line.strip().split('"')[1]  # 첫 번째 열 (입력값)
            data.append((input_string, line.strip(), last_col_value))
        # 마지막 열을 기준으로 데이터 정렬
        data.sort(key=lambda x: x[2])
    return data

# 명령어 실행 및 결과 파싱 함수
def run_command_and_parse_output(command):
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    output = result.stdout
    # 초기화
    top_1_accuracy = top_5_accuracy = "N/A"
    # 각 줄을 순회하며 필요한 정보 추출
    for line in output.split('\n'):
        if 'Inference acuracy' in line:
            # 줄을 쉼표로 나누고, 각 부분에서 정확도 값을 추출
            parts = line.split(',')
            top_1_part = parts[0]  # 'Inference accuracy: 67.50% (top-1'
            top_5_part = parts[2]  # ' 91.00% (top-5'

            # 공백으로 나누어 정확도 값 추출
            top_1_accuracy = top_1_part.split(' ')[2].strip('%')
            top_5_accuracy = top_5_part.split(' ')[1].strip('%')
            break

    return top_1_accuracy, top_5_accuracy

# 결과를 파일에 저장하는 함수
def save_results_to_file(file_path, input_string, last_col_value, top_1_accuracy, top_5_accuracy):
    with open(file_path, 'a') as file:
        file.write(f"{input_string}, {last_col_value}, Top-1 Accuracy: {top_1_accuracy}%, Top-5 Accuracy: {top_5_accuracy}%\n")

# 원본 데이터 파일 경로 및 결과 파일 경로
source_file_path = '/root/hetero-neurosim-search/Inference_pytorch/search_result/Mobilenet_V2_hetero/final_result_700_3_1_[1,1,1,1]_2_cka.txt'
results_file_path = 'inference_results_cka.txt'

# 파일 읽기 및 데이터 정렬
sorted_data = read_and_sort_file(source_file_path)




# 각 데이터에 대해 명령어 실행 및 결과 처리
ct = 0
for input_string, full_line, last_col_value in sorted_data:
    print(f"{ct}/{len(sorted_data)}")
    command = f"python -u run_inference_validate.py --model=MobileNetV2 --ntest=200 --ntest_batch=200 --input=\"{input_string}\""
    top_1_accuracy, top_5_accuracy = run_command_and_parse_output(command)
    save_results_to_file(results_file_path, input_string, last_col_value, top_1_accuracy, top_5_accuracy)
    ct += 1