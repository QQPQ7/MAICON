# MAICON

## 과제개요

* 항공 이미지 데이터를 활용한 건물 변화 탐지 
  * 이미지 영역 | 항공 이미지 데이터를 활용한 건물 변화 탐지 | mIoU


* 문제 정의 
  * 전후 이미지를 비교하여 건물 변화를 탐지 및 분할하는 문제


## 평가지표

*리더보드

- 경진대회 평가용 데이터는 Public과 Private 데이터로 이루어짐 (어떤 데이터가 Public/Private인지 참가자에게 공개하지 않음)

- 대회 기간 중에는 Public 데이터를 기반으로 Public 리더보드를 공개함

- Public 리더보드에 반영되는 제출 파일은 팀의 모든 제출 건 중 가장 높은 순위의 파일이 자동으로 선택됨

- 대회 종료 후 Private 리더보드와 Public과 Private 점수를 3:7의 비율로 합산하여 산출한 Final 리더보드를 공개함  

- 대회 마감 전 Final 리더보드에 반영할 제출 파일 1개를 최종 선택해야 함 (미선택 시 Public 리더보드 기준 최고점 파일 자동 반영) 

- 최종 순위는 Final 리더보드와 상이할 수 있으며 사후 검증 절차 후 확정됨


평가지표 : mIoU (mean Intersection over Union)


* 평가 로직은 keras.metrics.MeanIoU와 동일


## 데이터 개요
전후 비교 형태로 가공된 항공 이미지와 과거대비 변화가 있는 건물에 대한 Mask 이미지



*데이터 구성
- Train 데이터

  L x : 전후 비교 형태로 가공된 항공 이미지 (입력 정보, png 포맷)

  L y : 과거대비 변화가 있는 건물에 대한 세그멘테이션 Mask 이미지 (타겟 정보, png 포맷)


- Test 데이터 (Public - 30%, Private - 70% 포함)

  L x : 전후 비교 형태로 가공된 항공 이미지 (png 포맷)



*데이터 예시
- 파일 명칭 구조

--- train 데이터의 경우 이미지쌍연도(4자리)_자치구(3자리)_용도지역(3자리)_ID(6자리).png

ex) 2015_DMB_2LB_000286.png



--- test 데이터의 경우 ID(4자리).png

ex) 1000.png



- Mask 인코딩 설명

--- Pixel 별로 Background는 0, 신축은 1, 소멸은 2, 갱신은 3으로 인코딩되어 있음

--- Background (Pixel Value 0) : 신축, 소멸, 갱신 중 어느것에도 해당되지 않는 경우

--- 신축 (Pixel Value 1) : 왼쪽 대비 오른쪽 사진에 건물이 생성된 경우

--- 소멸 (Pixel Value 2): 왼쪽 대비 오른쪽 사진에서 건물이 붕괴되어 없는 경우

--- 갱신 (Pixel Value 3): 양쪽 같은 위치에 건물이 있으면서 한쪽에 변화가 있는 경우

![다운로드](https://user-images.githubusercontent.com/41661483/204326708-cef630bd-f919-4873-bdf7-84129c3cd567.png)


- (왼쪽 before, 오른쪽 after)

![다운로드 (1)](https://user-images.githubusercontent.com/41661483/204327439-a53def09-9d44-4abe-8788-36c42c1b7a80.png)


![신축예시](https://user-images.githubusercontent.com/41661483/204328222-19f4dcc5-15bf-4443-b828-d63a12d088e2.png)

<신축 예시 - 파란색>

![소멸예시](https://user-images.githubusercontent.com/41661483/204328242-211d2eb1-cca2-444c-8731-a8d856c51a31.png)

<소멸 예시 - 빨간색>

![갱신예시](https://user-images.githubusercontent.com/41661483/204328250-6e974e00-4b76-4594-afb9-e00d7d745629.png)

<갱신 예시 - 초록색>


## 사용한 네트워크 모델

 - unet

![unet](https://user-images.githubusercontent.com/41661483/204336692-1698c040-8fe5-4e5e-b50f-f404d7843223.png)

 - FPN (Feature Pyramid Networks)

![FPN_2](https://user-images.githubusercontent.com/41661483/204337669-55323973-2165-457c-9521-fe12e614af0b.png)

 
 

