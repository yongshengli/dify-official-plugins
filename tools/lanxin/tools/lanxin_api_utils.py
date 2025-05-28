import hashlib
import base64
import hmac
import time
from typing import Any, List, Optional, Dict, Tuple

def gen_sign(secret:str)->Tuple[str, str]:
    timestamp = int(round(time.time()))
    # print(timestamp)
    string_to_sign = '{}@{}'.format(timestamp, secret)
    # print(string_to_sign)
    hmac_code = hmac.new(string_to_sign.encode("utf-8"), digestmod=hashlib.sha256).digest()
    return base64.b64encode(hmac_code).decode('utf-8'), str(timestamp)