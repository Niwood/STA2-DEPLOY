import time



def secure_fetch(function):
    def wrapper(*args, **kwargs):

        fetched = False
        while not fetched:
            try:
                out = function(*args, **kwargs)
                fetched = True
                return out
            except:
                print(f'Not able to fetch, will retry in 2 seconds')
                time.sleep(2)

    return wrapper