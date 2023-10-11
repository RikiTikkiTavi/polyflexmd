import logging


def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(name)s :: %(message)s', datefmt='%d.%m.%Y %I:%M:%S'
    )
    logging.getLogger("fsspec.local").setLevel(logging.INFO)
    logging.getLogger("distributed.scheduler").setLevel(logging.WARNING)
    logging.getLogger("distributed.core").setLevel(logging.WARNING)
    logging.getLogger("distributed.nanny").setLevel(logging.WARNING)
    logging.getLogger("distributed.utils_perf").setLevel(logging.WARNING)
    logging.getLogger("tornado.application").setLevel(logging.CRITICAL)
    logging.getLogger("dask_jobqueue.core").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)
