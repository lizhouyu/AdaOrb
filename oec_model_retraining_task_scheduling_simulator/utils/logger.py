import logging

# logger
def getMyLogger(log_file='log.txt', mode='w'):
    logging.basicConfig(level=logging.INFO, 
        format="%(asctime)s %(name)s %(levelname)s %(lineno)d %(module)s %(message)s", 
        datefmt='%Y-%m-%d %H:%M:%S %a')
    logger = logging.getLogger()
    fh = logging.FileHandler(log_file, mode)
    formatter = logging.Formatter("%(asctime)s %(name)s %(processName)-10s %(levelname)s [%(filename)s %(module)s line: %(lineno)d] %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger