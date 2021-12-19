import argparse

def fun(args):
    print(args.a+args.b)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.a=1
    args.b=2
    fun(args)#这样使用的话，就能够直接args赋值给进去了