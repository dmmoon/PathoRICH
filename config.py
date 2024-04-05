# 클러스터 개수
N_CLUSTER = 5 

# Clustering Map Figure 크기 (Width, Height)
figsize = (7, 7) 

# 5x 패치 이미지에서 Cancer 패치 이미지 선별을 위한 임계값, 패치이미지에서 Cancer 영역이 포함하는 비율 설정 (0 ~ 1, e.g. 0.3 = Cancer 영역이 30% 이상일 경우 Cancer 패치로 분류)
LOWSCALE_CANCER_THRESHOLD = 0.2 

# 20x 패치 이미지에서 Cancer 패치 이미지 선별을 위한 임계값, 패치이미지에서 Cancer 영역이 포함하는 비율 설정 (0 ~ 1, e.g. 0.5 = Cancer 영역이 50% 이상일 경우 Cancer 패치로 분류)
HIGHSCALE_CANCER_THRESHOLD = 0.5

# 상위/하위 attention score N% 패치 이미지 추출 위한 값, e.g. 0.05 = attention score 상위/하위 5% 패치 이미지 추출
TOP_SCORE_RATIO = 0.2
TARGET_SPACING = 0.5041

THRESHOLD_20x = [
                 0.6670,
                 0.3332
                 ]
THRESHOLD_5x = [0.6873, 0.3142]