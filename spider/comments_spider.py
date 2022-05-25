# coding=utf-8
from importlib import reload

import requests
from bs4 import BeautifulSoup
import sys
import codecs
import time
import random
import csv
from ua_info import ua_list


def spider_comment(percent_type):
    # 《敦刻尔克》豆瓣评论地址
    url = 'https://movie.douban.com/subject/26607693/comments?'+percent_type +'&'+'start=0'

    head = {'User-Agent': random.choice(ua_list)}
    cookies = {'cookie': 'bid=ewWPeAHwNlk; douban-fav-remind=1; __yadk_uid=CAAqrzXU0V12lAnDZeP4oiMIRMHApwgc; __gads=ID=230a6ca70ae465d2-2276fd16b6cb0014:T=1632038483:RT=1632038483:S=ALNI_MaCb5AjlpkVt8xRpkAlefSXOfFmsQ; ll="118281"; __gpi=UID=000005982219702e:T=1653027335:RT=1653027335:S=ALNI_MZHo6DCjVbPJVb00IgSGgJZn-LgGg; ct=y; __utmc=30149280; __utmz=30149280.1653464479.3.2.utmcsr=baidu|utmccn=(organic)|utmcmd=organic; __utma=30149280.657379030.1633594810.1653464479.1653467359.4; apiKey=; dbcl2="211693886:Cde/oWu/iAU"; ck=nyWa; _pk_ref.100001.8cb4=["","",1653475185,"https://accounts.douban.com/"]; _pk_ses.100001.8cb4=*; ap_v=0,6.0; push_doumail_num=0; push_noty_num=0; _pk_id.100001.8cb4=ccbcbab9e392eadf.1632038481.6.1653475204.1641895579.'}

    f = open('dbcomments.csv', 'a', newline='', encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(['日期', '评星', '赞成数', '评论内容'])

    # 请求网站
    html = requests.get(url, headers=head, cookies=cookies)

    while html.status_code == 200:

        # 生成BeautifulSoup对象
        soup = BeautifulSoup(html.text, 'html.parser')
        comment = soup.find_all(class_='comment')

        # 解析每一个class为comment的div中的内容
        for com in comment:

            # 评论内容
            comments = com.find(class_='short')
            commentstring = comments.string
            # print(commentstring)

            # 评星(会出现没有评星的情况,没有评星设置为None)
            rating = com.find(class_='rating')
            if rating != None:
                ratingstring = rating.get('title')
            else:
                ratingstring = 'None'
            # print(ratingstring)

            # 评论有用的数量
            votes = com.find(class_='votes')
            votesstring = votes.string
            # print(votesstring)


            # 日期
            commenttime = com.find(class_='comment-time')
            timestring = commenttime.get('title')
            # print (timestring)

            # 写入csv文件一行数据
            try:
                writer.writerow([timestring, ratingstring, votesstring, commentstring])
            except Exception as err:
                print(err)

        time.sleep(3)
        # 下一页
        nextstring = soup.find(class_='next').get('href')
        print(nextstring)
        nexturl = 'https://movie.douban.com/subject/26607693/comments' + nextstring
        print(html.status_code)
        html = requests.get(nexturl, headers=head, cookies=cookies)

    f.close()

if __name__ == '__main__':
    # 好评
    percent_type = 'percent_type=h'
    spider_comment(percent_type)
    # 差评
    percent_type = 'percent_type=l'
    spider_comment(percent_type)