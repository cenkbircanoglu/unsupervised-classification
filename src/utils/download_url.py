from urllib.request import urlretrieve

from tqdm import tqdm


def download_url(url, destination=None, progress_bar=True):
    """Download a URL to a local file.

    Parameters
    ----------
    url : str
        The URL to download.
    destination : str, None
        The destination of the file. If None is given the file is saved to a temporary directory.
    progress_bar : bool
        Whether to show a command-line progress bar while downloading.

    Returns
    -------
    filename : str
        The location of the downloaded file.

    Notes
    -----
    Progress bar use/example adapted from tqdm documentation: https://github.com/tqdm/tqdm
    """

    def progressbar_hook(t):
        last_b = [0]

        def inner(b=1, bsize=1, tsize=None):
            if tsize is not None:
                t.total = tsize
            if b > 0:
                t.update((b - last_b[0]) * bsize)
            last_b[0] = b

        return inner

    if progress_bar:
        with tqdm(unit='B',
                  unit_scale=True,
                  miniters=1,
                  desc=url.split('/')[-1]) as t:
            filename, _ = urlretrieve(url,
                                      filename=destination,
                                      reporthook=progressbar_hook(t))
    else:
        filename, _ = urlretrieve(url, filename=destination)
    return filename