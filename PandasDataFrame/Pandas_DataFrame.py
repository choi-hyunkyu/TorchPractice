# 데이터프레임 데이터 불러오기
df = pd.read_csv('./PATH/file_name.csv')

# 데이터프레임 데이터 컬럼 인덱스 제외
df = pd.read_csv('./PATH/file_name.csv', index_col = [0])

# 데이터프레임 데이터 컬럼 인덱스 부여
df = df.set_index('column_name')
df.columns = ['column_name']

# 데이터프레임 데이터 인덱스 제거
df = df.reset_index(drop = True)
df.reset_index(drop = True, inplace = True)

# 데이터프레임 컬럼 추가
df['NewColumn_name'] = 'Values'

# 데이터프레임 컬럼 이름 변경
df = df.rename(columns = {'old_name_1': 'new_name_1', 'old_name_2': 'new_name_2'})
df.columns = ['new_name_1', 'new_name_2']

# 데이터프레임 컬럼 삭제
df = df.drop(['name_1', 'name_2'], axis = 1)
df.drop(['name_1', 'name_2'], axis = 1, replace = True)

# 데이터프레임 데이터타입 변환
df = df.astype('float')
df.astype('float', replace = True)

# 날짜를 사용할 때 유형을 datetime으로 변경
df = pd.to_datetime(df['Moth'])

# 날짜 datetime 타입으로 변환
original_data_date_list = original_data_df['Month'].tolist()
original_data_date_list = pd.to_datetime(original_data_date_list)
original_data_df['Month'] = original_data_date_list
print(original_data_df.head())
type(original_data_df['Month'][0])

# Month 컬럼 인덱스 부여
original_data_df = original_data_df.set_index('Month').astype(int) # Month 컬럼에 인덱스 부여
original_data_df = original_data_df[:1000]
print(original_data_df.head())

# 데이터 셔플
shuffled_df = df.sample(frac = 1)

# 결측치 0으로 치환
filled_df = df.fillna(0)

# 결측치 특정 문자열로 치환
filled_df = df.fillna("string")

# 결측치 앞 방향 값으로 치환
filled_df = df.fillna('ffill')
filled_df = df.fillna('pad')

# 결측치 뒷 방향 값으로 치환
filled_df = df.fillna('bfill')
filled_df = df.fillna('backfill')

# 결측치 변수별 평균으로 대체
filled_df = df.fillna(df.mean())
