# 1. 베이스 이미지 선택
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# 2. 작업 디렉토리 설정
# 컨테이너 내부에서 명령어가 실행될 기본 경로를 지정합니다.
WORKDIR /app

# 3. 파이썬 의존성 설치 (매우 중요!)
# requirements.txt 파일을 먼저 복사하여 설치합니다.
# 이렇게 하면 소스 코드가 변경되어도 매번 라이브러리를 새로 설치하지 않고
# Docker의 캐시를 활용하여 빌드 속도가 매우 빨라집니다.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 프로젝트 소스 코드 복사
# 현재 폴더의 모든 파일을 컨테이너의 /app 폴더로 복사합니다.
COPY . .

# 5. 컨테이너 실행 시 기본 명령어 설정
# 컨테이너가 시작될 때 자동으로 bash 셸을 실행하도록 설정합니다.
# 이를 통해 컨테이너에 접속하여 자유롭게 명령어를 입력할 수 있습니다.
CMD ["bash"]