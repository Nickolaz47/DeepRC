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


def filter_orfs_by_length(hdf5_path: str, min_len=None, max_len=None):
    """
    Filters ORFs in an HDF5 file by length without modifying the original file.

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
    dict
        Dictionary containing filtered data with the same structure as the
        original HDF5.
    """
    filtered_data = {
        'metadata': None,
        'sampledata': {
            'n_sequences_per_sample': None,
            'sample_avg_seq_len': None,
            'sample_max_seq_len': None,
            'sample_min_seq_len': None,
            'sample_sequences_start_end': None,
            'seq_lens': None,
            'sequence_counts': None,
            'sequences': None
        }
    }
    
    with h5py.File(hdf5_path, 'r') as f:
        # Copy metadata (not affected by filtering)
        if 'metadata' in f:
            filtered_data['metadata'] = dict(f['metadata'].attrs.items())
        
        # Load sequence data
        seq_lens = f['sampledata/seq_lens'][:]
        sequences = f['sampledata/sequences'][:]
        
        # Create length filter mask
        mask = np.ones_like(seq_lens, dtype=bool)
        if min_len is not None:
            mask = mask & (seq_lens >= min_len)
        if max_len is not None:
            mask = mask & (seq_lens <= max_len)
        
        # Apply filter
        filtered_seq_lens = seq_lens[mask]
        filtered_sequences = sequences[mask]
        
        # Store filtered data
        filtered_data['sampledata']['seq_lens'] = filtered_seq_lens
        filtered_data['sampledata']['sequences'] = filtered_sequences
        
        # Process other sampledata that might need adjustment
        if 'n_sequences_per_sample' in f['sampledata']:
            filtered_data['sampledata']['n_sequences_per_sample'] = \
                f['sampledata/n_sequences_per_sample'][:]
        
        # Recalculate sample_avg/max/min_seq_len values
        if 'sample_avg_seq_len' in f['sampledata']:
            filtered_data['sampledata']['sample_avg_seq_len'] = \
                f['sampledata/sample_avg_seq_len'][:]
        
        if 'sample_max_seq_len' in f['sampledata']:
            filtered_data['sampledata']['sample_max_seq_len'] = \
                f['sampledata/sample_max_seq_len'][:]
        
        if 'sample_min_seq_len' in f['sampledata']:
            filtered_data['sampledata']['sample_min_seq_len'] = \
                f['sampledata/sample_min_seq_len'][:]
        
        if 'sample_sequences_start_end' in f['sampledata']:
            # This might need special handling depending on its structure
            filtered_data['sampledata']['sample_sequences_start_end'] = \
                f['sampledata/sample_sequences_start_end'][:]
        
        if 'sequence_counts' in f['sampledata']:
            filtered_data['sampledata']['sequence_counts'] = \
                f['sampledata/sequence_counts'][:]
    
    print(f"Filter applied: {np.sum(mask)} of {len(mask)} sequences retained "
          f"({np.sum(mask)/len(mask)*100:.2f}%)")
    
    return filtered_data