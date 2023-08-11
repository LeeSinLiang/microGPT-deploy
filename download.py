import tqdm
import functools
import urllib
from retry import retry


@functools.lru_cache()
def create_download_progress_bar():
    class DownloadProgressBar(tqdm.tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    return DownloadProgressBar


@retry((urllib.error.HTTPError, ConnectionResetError))
def download_with_progress(url, filepath):
    DownloadProgressBar = create_download_progress_bar()

    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
    ) as t:
        urllib.request.urlretrieve(url, filepath, reporthook=t.update_to)