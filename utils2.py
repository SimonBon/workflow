from time import time

def timeit(rep):
    def do_time(func):
        def wrap_func(*args, **kwargs):
            times = []
            for _ in range(rep):
                t1 = time()
                func(*args, **kwargs)
                times.append(time()-t1)
            print(f'Function {func.__name__!r} executed in {sum(times)/1000:.10f}s')
        return wrap_func
    return do_time