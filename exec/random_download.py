# Modules
# API requests
import requests
import urllib.request
from bs4 import BeautifulSoup

# data
import pandas as pd
import numpy as np
import re

# time
from datetime import datetime

# etc
import os
import argparse

# argparse
parser = argparse.ArgumentParser()
parser.add_argument('--auth_key', help='발급된 인증키', type=str)
parser.add_argument('--download_path', help='다운로드 경로', type=str)
parser.add_argument('--download_num', help='다운로드 개수', type=int, default=1000)
parser.add_argument('--road_type', help='ex(고속도로) / its(국도)', type=str, default='ex')
args = parser.parse_args()


def authentication(auth_key, road_type):
    if road_type == 'ex':
        url = 'http://openapi.its.go.kr:8081/api/NCCTVInfo?key={}&ReqType=2&MinX=120&MaxX=135&MinY=30&MaxY=45&type=ex'.format(auth_key) # 고속도로 url
    else:
        url = 'http://openapi.its.go.kr:8081/api/NCCTVInfo?key={}&ReqType=2&MinX=120&MaxX=135&MinY=30&MaxY=45&type=its'.format(auth_key) # 국도 url
    return url


def request_api(url):
    req = requests.get(url)
    html = req.text
    soup = BeautifulSoup(html, 'html.parser')

    # cctv 이름 / url / 좌표 데이터프레임
    name = [i.text for i in soup.findAll("cctvname")]
    url = [i.text for i in soup.findAll("cctvurl")]
    coordx = [i.text for i in soup.findAll("coordx")]
    coordy = [i.text for i in soup.findAll("coordy")]

    cctv_df = pd.DataFrame(data={'name':name, 'url':url, 'coordx':coordx, 'coordy':coordy})
    return cctv_df


# 파일로 저장될 수 없는 이름 수정
def format_name(cctv_name):
    p = re.compile('[\w\[\]]+')
    regex_name = re.findall(p, cctv_name)
    formatted_name = ''
    for i in range(len(regex_name)):
        formatted_name += regex_name[i]

    now = datetime.now()
    formatted_name = '{}_{}_{}'.format(now.strftime('%m%d'), now.strftime('%H%M'), formatted_name)
    return formatted_name


def random_select(cctv_df, download_num):
    random_idx = np.random.randint(low=0, high=len(cctv_df), size=download_num)
    random_cctv = cctv_df.loc[random_idx]
    random_cctv.loc[:,'formatted_name'] = random_cctv['name'].apply(lambda x: format_name(x))
    return random_cctv


def download_cctv(random_cctv, download_path):
    # 디렉토리 존재여부 확인
    if not os.path.isdir(download_path):
        os.mkdir(download_path)

    for i in range(len(random_cctv)):
        cctv = random_cctv.iloc[i]
        #urllib.request.urlretrieve(cctv['url'], '{}{}.mp4'.format(download_path, cctv['formatted_name']))
        try:
            urllib.request.urlretrieve(cctv['url'], os.path.join(download_path, '{}.mp4'.format(cctv['formatted_name'])))
        except:
            continue
    print('다운로드 완료')


def main(args):
    url = authentication(args.auth_key, args.road_type)
    cctv_df = request_api(url)
    random_cctv = random_select(cctv_df, args.download_num)
    download_cctv(random_cctv, args.download_path)


if __name__ == '__main__':
    main(args)
