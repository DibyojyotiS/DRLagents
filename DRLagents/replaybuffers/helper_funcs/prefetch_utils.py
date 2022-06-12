from queue import Queue
from threading import Thread
import time

def threading_prefetch_wrapper(func, nprefetch=0, nthreads=1):
    """ To prefetch somethings in advance using 
    multi-threading daemon threads. \\
    The threads are started only after the first 
    call to the wrapped func.

    ### parameters
    1. func: function
            - this is the function that will 
            return stuff to prefetch. Will be
            repeatedly called by threads untill 
            the parent thread ends.
    2. nprefetch: int (default 0)
            - set to > 0 to get prefetching behaviour
            - the size of the prefetch queue
            - when the queue is full, each thread
            would possibly be waiting to put the next
            result into the queue.
    3. nthreads: int (default 5)
            - the number of daemon threads 

    ### returns 
        - wrapped function """

    if nprefetch == 0: return func

    threads_started = False
    prefetch_queue = Queue(maxsize=nprefetch)

    # function to init and start thread
    def start_threads(*args, **kwargs):
        nonlocal threads_started, prefetch_queue
        threads_started = True
        prefetch_threads = [Thread(target=worker, args=(i,*args), 
                kwargs=kwargs) for i in range(nthreads)]
        for thread in prefetch_threads: thread.setDaemon(True)
        for thread in prefetch_threads: thread.start(),time.sleep(0.01)

    # the threading worker
    def worker(i, *args, **kwargs):
        print(f'started prefetch thread {i}')
        while True:
            sample = func(*args, **kwargs)
            prefetch_queue.put(sample)

    # the threading fetcher (start threads on 1st call)
    def fetcher(*args, **kwargs):
        if not threads_started: start_threads(*args, **kwargs)
        # print(prefetch_queue.qsize())
        return prefetch_queue.get()

    return fetcher

def threading_prefetch_decorator(nprefetch=0, nthreads=1):
    """ a decorator to prefetch stuffs using 
    multiple daemon threads. \\
    The threads are started only after the first
    call to the decorated function.
    ### parameters
    1. nprefetch: int (default 0)
            - set to > 0 to get prefetching behaviour
            - the size of the prefetch queue
            - when the queue is full, each thread
            would possibly be waiting to put the next
            result into the queue.
    2. nthreads: int (default 5)
            - the number of daemon threads  """
    def wrapper(func):
        return threading_prefetch_wrapper(func, nprefetch, nthreads)
    return wrapper