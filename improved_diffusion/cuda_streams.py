from functools import wraps
import torch as th

_MAIN_STREAM = th.cuda.Stream()
_TXT_STREAM = th.cuda.Stream()
_CAPT_STREAM = th.cuda.Stream()

_EMB_STREAM = th.cuda.Stream()
_CONV_STREAM = th.cuda.Stream()

_GLOBAL_FLAGS = {"streaming_on": False}


def is_streaming_on():
    return _GLOBAL_FLAGS["streaming_on"]


def turn_streaming_on():
    _GLOBAL_FLAGS["streaming_on"] = True


def turn_streaming_off():
    _GLOBAL_FLAGS["streaming_on"] = False


def main():
    if is_streaming_on():
        return _MAIN_STREAM
    return th.cuda.default_stream()


def txt():
    if is_streaming_on():
        return _TXT_STREAM
    return th.cuda.default_stream()


def capt():
    if is_streaming_on():
        return _CAPT_STREAM
    return th.cuda.default_stream()


def emb():
    if is_streaming_on():
        return _EMB_STREAM
    return th.cuda.default_stream()


def conb():
    if is_streaming_on():
        return _CONV_STREAM
    return th.cuda.default_stream()


def use_main_stream(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        with th.cuda.stream(main()):
            return f(*args, **kwargs)
    return wrapper
