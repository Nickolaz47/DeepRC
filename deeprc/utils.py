# -*- coding: utf-8 -*-
"""
Utility functions and classes

Author -- Michael Widrich
Contact -- widrich@ml.jku.at
"""
import os
import requests
import shutil
import tqdm
import h5py
import numpy as np


def user_confirmation(text: str = "Continue?", continue_if: str = 'y', abort_if: str = 'n'):
    """Wait for user confirmation"""
    while True:
        user_input = input(f"{text} ({continue_if}/{abort_if})")
        if user_input == continue_if:
            break
        elif user_input == abort_if:
            exit("Session terminated by user.")


def url_get(url: str, dst: str, verbose: bool = True):
    """Download file from `url` to file `dst`"""
    stream = requests.get(url, stream=True)
    try:
        stream_size = int(stream.headers['Content-Length'])
    except KeyError:
        raise FileNotFoundError(f"Sorry, the URL {url} could not be reached. "
                                f"Either your connection has a problem or the server is down."
                                f"Please check your connection, try again later, "
                                f"or notify me per email if the problem persists.")
    src = stream.raw
    windows = os.name == 'nt'
    copy_bufsize = 1024 * 1024 if windows else 64 * 1024
    update_progess_bar = tqdm.tqdm(total=stream_size, disable=not verbose,
                                   desc=f"Downloading {stream_size * 1e-9:0.3f}GB dataset")
    with open(dst, 'wb') as out_file:
        while True:
            buf = src.read(copy_bufsize)
            if not buf:
                break
            update_progess_bar.update(copy_bufsize)
            out_file.write(buf)
        shutil.copyfileobj(stream.raw, out_file)
    print()
    del stream


def filter_orfs_by_length(hdf5_path: str, min_len=None, max_len=None) -> str:
    """
    Filters ORFs in an HDF5 file by length and writes to a new HDF5 file.

    Parameters
    ----------
    hdf5_path : str
        Path to the HDF5 file containing the data.
    min_len : int, optional
        Minimum length of ORFs to keep. If None, no minimum filter is applied.
    max_len : int, optional
        Maximum length of ORFs to keep. If None, no maximum filter is applied.

    Returns
    -------
    str
        Path to the new HDF5 file containing filtered data.
    """
    with h5py.File(hdf5_path, "r") as f:
        # Create output file path
        dirname = os.path.dirname(hdf5_path)
        basename = os.path.basename(hdf5_path)
        output_path = os.path.join(
            dirname,
            f"{os.path.splitext(basename)[0]}_filtered_{min_len}_{max_len}.hdf5"
        )
        
        # Create mask for filtering
        seq_lens = f["sampledata/seq_lens"][:]
        mask = np.ones_like(seq_lens, dtype=bool)
        if min_len is not None:
            mask = mask & (seq_lens >= min_len)
        if max_len is not None:
            mask = mask & (seq_lens <= max_len)
        
        # Create new HDF5 file with filtered data
        with h5py.File(output_path, "w") as out:
            # Copy metadata
            if "metadata" in f:
                for key, value in f["metadata"].attrs.items():
                    out["metadata"].attrs[key] = value
            
            # Create sampledata group
            sampledata = out.create_group("sampledata")
            
            # Write filtered sequences and lengths
            sampledata.create_dataset("sequences", data=f["sampledata/sequences"][:][mask])
            sampledata.create_dataset("seq_lens", data=seq_lens[mask])
            
            # Copy other datasets that don't need filtering
            for dataset_name in [
                "n_sequences_per_sample",
                "sample_sequences_start_end",
                "sequence_counts"
            ]:
                if dataset_name in f["sampledata"]:
                    f.copy(f"sampledata/{dataset_name}", sampledata)
            
            # Recalculate and write length statistics
            if all(name in f["sampledata"] for name in ["sample_avg_seq_len", "sample_max_seq_len", "sample_min_seq_len"]):
                # Get original sample indices and counts
                sample_counts = f["sampledata/n_sequences_per_sample"][:]
                cum_counts = np.cumsum(sample_counts)
                starts = np.concatenate(([0], cum_counts[:-1]))
                
                # Initialize new stats arrays
                new_avg = np.zeros_like(f["sampledata/sample_avg_seq_len"][:])
                new_max = np.zeros_like(f["sampledata/sample_max_seq_len"][:])
                new_min = np.zeros_like(f["sampledata/sample_min_seq_len"][:])
                
                # Calculate new stats per sample
                for i, (start, count) in enumerate(zip(starts, sample_counts)):
                    sample_mask = mask[start:start+count]
                    if np.any(sample_mask):
                        sample_lens = seq_lens[start:start+count][sample_mask]
                        new_avg[i] = np.mean(sample_lens)
                        new_max[i] = np.max(sample_lens)
                        new_min[i] = np.min(sample_lens)
                    else:
                        new_avg[i] = new_max[i] = new_min[i] = 0
                
                sampledata.create_dataset("sample_avg_seq_len", data=new_avg)
                sampledata.create_dataset("sample_max_seq_len", data=new_max)
                sampledata.create_dataset("sample_min_seq_len", data=new_min)
    
    print(f"Filter applied: {np.sum(mask)} of {len(mask)} sequences retained "
          f"({np.sum(mask)/len(mask)*100:.2f}%)")
    
    return output_path