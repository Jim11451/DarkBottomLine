#!/usr/bin/env python3
"""
Script to merge multiple pickle files from DarkBottomLine outputs.
"""

import pickle
import glob
import os
import argparse


def merge_pkl_files(folder_path,chunk, output_file='merged.pkl'):
    """
    Merge all pkl files in the specified folder.
    
    Args:
        folder_path: Path to the folder containing pkl files
        output_file: Name of the output merged file
    """
    global pkl_files
    if not pkl_files:
        print('No pkl files found')
        return
    
    # Initialize merged data
    merged_events = []
    merged_objects = {}
    first = True
    start = (chunk-1)*100
    end = chunk*100
    for i, pkl_file in enumerate(pkl_files[start:end]):
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            # Merge events
            if 'events' in data:
                merged_events.extend(data['events'])
            
            # Merge objects
            if 'objects' in data:
                if first:
                    merged_objects = {k: [] for k in data['objects']}
                    first = False
                for k in merged_objects:
                    if k in data['objects']:
                        merged_objects[k].extend(data['objects'][k])
            
            if (i + 1) % 100 == 0:
                print(f'Processed {i + 1} files')
                
        except Exception as e:
            print(f'Error processing {pkl_file}: {e}')
            continue
    
    # Save merged data
    merged_data = {
        'events': merged_events,
        'objects': merged_objects
    }
    
    output_path = os.path.join(folder_path, output_file)
    with open(output_path, 'wb') as f:
        pickle.dump(merged_data, f)
    
    print(f'Merged data saved to {output_path}')
    # print(f'Total events: {len(merged_events)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge pkl files in a specified folder.')
    parser.add_argument('folder', help='Path to the folder containing pkl files')
    parser.add_argument('--output', default='merged.pkl', help='Name of the output file (default: merged.pkl)')
    
    args = parser.parse_args()
    pkl_files = glob.glob(os.path.join(args.folder, '*.pkl'))
    pkl_files = [f for f in pkl_files if not f.endswith('.awk_raw.pkl')]
    print(f'Found {len(pkl_files)} pkl files to merge')
    if not os.path.isdir(args.folder):
        print(f'Error: {args.folder} is not a valid directory')
        exit(1)
    for chunk in range(1,len(pkl_files)//100+2):
        output = args.output + f'{chunk}' + '.pkl'
        merge_pkl_files(args.folder,chunk, output)
