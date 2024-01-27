import os
import random

import numpy as np
import requests
from bs4 import BeautifulSoup
from keras import models
from keras.src.utils import load_img, img_to_array


def split_digits_in_img(img_array):
    x_list = list()
    for i in range(digits_in_img):
        step = img_cols // digits_in_img
        x_list.append(img_array[:, i * step:(i + 1) * step] / 255)
    return x_list


def get_img_src(session: requests.Session) -> str:
    res = session.get(BASE_URL)
    soup = BeautifulSoup(res.text, 'lxml')
    img = soup.find('form').find('img')
    return img['src']


BASE_URL = 'https://www.ccxp.nthu.edu.tw/ccxp/INQUIRE/'
times = 1000

for run_acc in range(0, times):
    sess = requests.Session()
    pwd = get_img_src(sess)
    res = sess.get(BASE_URL + pwd)
    with open('for_predict/temp.png', 'wb') as f:
        f.write(res.content)

    img_rows = 80
    img_cols = 30
    digits_in_img = 6
    model = None
    np.set_printoptions(suppress=True, linewidth=150, precision=9, formatter={'float': '{: 0.9f}'.format})

    if os.path.isfile('cnn_model.keras'):
        model = models.load_model('cnn_model.keras')
    else:
        print('No trained model found.')
        exit(-1)

    img = load_img('for_predict/temp.png', color_mode='grayscale').resize(size=(img_rows, img_cols))
    img_array = img_to_array(img)
    img_rows, img_cols, _ = img_array.shape
    x_list = split_digits_in_img(img_array)

    varification_code = list()
    for i in range(digits_in_img):
        confidences = model.predict(np.array([x_list[i]]), verbose=0)
        result_class = np.argmax(confidences, axis=-1)
        varification_code.append(result_class[0])

    predict_code = str().join(str(i) for i in varification_code)
    print('#{0} Predict: {1}'.format(run_acc, predict_code))

    ans = sess.post(BASE_URL + 'pre_select_entry.php', data={
        "account": random.randint(0, 112000000),
        "passwd": "password",
        "passwd2": predict_code,
        "Submit": "提交",
        "fnstr": pwd[pwd.index("=") + 1:]
    })
    ans.encoding = 'big5'
    print(ans.text)
    if "select_entry.php" not in ans.text:
        if "15分鐘內登錄錯誤超過5次" in ans.text:
            print('Wrong prediction for 5 times.')
            break
        print('Not {0}! Store in wrong_prediction/'.format(predict_code))
        with open('wrong_prediction/{0}_{1}.png'.format(predict_code, random.randint(0, 100000000)), 'wb') as f:
            f.write(res.content)
    else:
        with open('gen_data/{0}_{1}.png'.format(predict_code, random.randint(0, 100000000)), 'wb') as f:
            f.write(res.content)

    sess.close()
