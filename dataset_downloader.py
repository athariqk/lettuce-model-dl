import os
import json
import requests
import time
import aiohttp
import asyncio
from tqdm import tqdm
import subprocess
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import io

# Configuration - adjust based on your network and system capabilities
MAX_CONCURRENT_DOWNLOADS = 50
MAX_CONCURRENT_VALIDATIONS = 100
CHUNK_SIZE = 1024 * 16
CONNECTION_TIMEOUT = 10
RETRY_ATTEMPTS = 3
REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return process.returncode, stdout, stderr

async def download_file_async(session, url, destination, semaphore, pbar=None):
    """Download a single file asynchronously with retries"""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            async with semaphore:  # Limit concurrency
                async with session.get(url, headers=REQUEST_HEADERS, timeout=CONNECTION_TIMEOUT) as response:
                    if response.status != 200:
                        if attempt < RETRY_ATTEMPTS - 1:
                            # Exponential backoff
                            await asyncio.sleep(1 * (2 ** attempt))
                            continue
                        return False, f"Failed to download {url}, status: {response.status}"
                    
                    # Make sure directory exists
                    os.makedirs(os.path.dirname(destination), exist_ok=True)
                    
                    # Download and write file
                    with open(destination, 'wb') as f:
                        async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                            f.write(chunk)
                    
                    # Verify the downloaded image is valid
                    if not is_valid_image(destination):
                        if attempt < RETRY_ATTEMPTS - 1:
                            # If image is corrupted, try again
                            continue
                        return False, f"Downloaded file {url} is corrupted"
                    
                    if pbar:
                        pbar.update(1)
                    return True, destination
        except Exception as e:
            if attempt < RETRY_ATTEMPTS - 1:
                # Exponential backoff
                await asyncio.sleep(1 * (2 ** attempt))
                continue
            return False, f"Error downloading {url}: {str(e)}"
    
    return False, f"Failed to download {url} after {RETRY_ATTEMPTS} attempts"

def is_valid_image(file_path):
    """Check if an image file is valid (not corrupted)"""
    try:
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return False
        
        # Try to open and verify the image
        with Image.open(file_path) as img:
            # Try to load the image data which forces PIL to decode and verify
            img.verify()
            return True
    except Exception:
        # Any exception indicates a corrupt or invalid image
        if os.path.exists(file_path):
            try:
                os.remove(file_path)  # Remove corrupted file
            except:
                pass
        return False

async def validate_image_async(file_info, semaphore, pbar=None):
    """Validate a single image file asynchronously"""
    async with semaphore:
        valid = is_valid_image(file_info['path'])
        if pbar:
            pbar.update(1)
        if not valid:
            return False, file_info
        return True, file_info

async def validate_images_batch_async(files_to_check):
    """Validate a batch of image files asynchronously"""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_VALIDATIONS)
    
    pbar = tqdm(total=len(files_to_check), desc="Validating images", unit="file")
    tasks = []
    
    for file_info in files_to_check:
        task = validate_image_async(file_info, semaphore, pbar)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    pbar.close()
    
    # Collect invalid files
    invalid_files = [file_info for valid, file_info in results if not valid]
    return invalid_files

async def download_multiple_files_async(url_dest_pairs):
    """Download multiple files concurrently with async/await"""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
    timeout = aiohttp.ClientTimeout(total=None, connect=CONNECTION_TIMEOUT)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        pbar = tqdm(total=len(url_dest_pairs), desc="Downloading files", unit="file")
        tasks = []
        
        for url, dest in url_dest_pairs:
            task = download_file_async(session, url, dest, semaphore, pbar)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        pbar.close()
        
        # Process results
        successes = [r for r in results if isinstance(r, tuple) and r[0]]
        failures = [r for r in results if isinstance(r, tuple) and not r[0] or not isinstance(r, tuple)]
        
        return successes, failures

def download_large_file(url, destination):
    """Download a single large file with progress indication"""
    try:
        with requests.get(url, stream=True, headers=REQUEST_HEADERS, timeout=CONNECTION_TIMEOUT) as response:
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            
            with open(destination, 'wb') as f, tqdm(
                total=total_size, unit='B', unit_scale=True, desc=os.path.basename(destination)
            ) as pbar:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            # Verify integrity of the downloaded file
            if '.zip' in destination:
                # For zip files, a simple existence check is sufficient
                if os.path.exists(destination) and os.path.getsize(destination) > 0:
                    return True
                return False
            
            return True
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False

def check_file_existence_batch(files_to_check, batch_size=5000):
    """Check file existence in batches to avoid memory issues with large datasets"""
    missing_files = []
    batches = [files_to_check[i:i + batch_size] for i in range(0, len(files_to_check), batch_size)]
    
    for batch_idx, batch in enumerate(batches):
        print(f"Checking batch {batch_idx+1}/{len(batches)}...")
        for file_info in tqdm(batch):
            file_path = file_info['path']
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                missing_files.append(file_info)
    
    return missing_files

async def check_and_download_coco_async(coco_root):
    """Main function to check and download COCO dataset with async support"""
    os.makedirs(coco_root, exist_ok=True)
    
    # Define directories
    train_dir = os.path.join(coco_root, 'train2017')
    val_dir = os.path.join(coco_root, 'val2017')
    annotations_dir = os.path.join(coco_root, 'annotations')
    
    # Create directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)
    
    # Download large zip files if needed
    large_files_to_check = [
        {
            'name': 'COCO 2017 train images',
            'dir': train_dir,
            'url': 'http://images.cocodataset.org/zips/train2017.zip',
            'zip_path': os.path.join(coco_root, 'train2017.zip'),
            'extract_dir': coco_root
        },
        {
            'name': 'COCO 2017 annotations',
            'file': os.path.join(annotations_dir, 'instances_train2017.json'),
            'url': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
            'zip_path': os.path.join(coco_root, 'annotations_trainval2017.zip'),
            'extract_dir': coco_root
        },
        {
            'name': 'COCO 2017 val images',
            'dir': val_dir,
            'url': 'http://images.cocodataset.org/zips/val2017.zip',
            'zip_path': os.path.join(coco_root, 'val2017.zip'),
            'extract_dir': coco_root
        }
    ]
    
    # Download and extract large files if needed
    for file_info in large_files_to_check:
        # Check if directory or file exists
        if 'dir' in file_info and os.path.exists(file_info['dir']) and len(os.listdir(file_info['dir'])) > 0:
            print(f"{file_info['name']} directory exists and is not empty.")
        elif 'file' in file_info and os.path.exists(file_info['file']):
            print(f"{file_info['name']} file exists.")
        else:
            print(f"Downloading {file_info['name']}...")
            if download_large_file(file_info['url'], file_info['zip_path']):
                print(f"Extracting {file_info['name']}...")
                run_command(f"unzip -q {file_info['zip_path']} -d {file_info['extract_dir']}")
                os.remove(file_info['zip_path'])
            else:
                print(f"Failed to download {file_info['name']}.")
    
    # Verify integrity of the dataset
    await verify_dataset_integrity_async(coco_root)
    
    print(f"MS COCO 2017 dataset processing completed at {coco_root}")

async def verify_dataset_integrity_async(coco_root):
    """Verify integrity of the dataset and download missing files asynchronously"""
    train_annotations_file = os.path.join(coco_root, 'annotations', 'instances_train2017.json')
    val_annotations_file = os.path.join(coco_root, 'annotations', 'instances_val2017.json')
    
    all_files_to_process = []
    
    # Load train images info
    if os.path.exists(train_annotations_file):
        print("Loading train annotations...")
        with open(train_annotations_file, 'r') as f:
            train_data = json.load(f)
        
        train_files = [
            {
                'path': os.path.join(coco_root, 'train2017', img['file_name']),
                'url': f"http://images.cocodataset.org/train2017/{img['file_name']}",
                'dest': os.path.join(coco_root, 'train2017', img['file_name']),
                'type': 'train'
            }
            for img in train_data['images']
        ]
        all_files_to_process.extend(train_files)
    
    # Load val images info
    if os.path.exists(val_annotations_file):
        print("Loading val annotations...")
        with open(val_annotations_file, 'r') as f:
            val_data = json.load(f)
        
        val_files = [
            {
                'path': os.path.join(coco_root, 'val2017', img['file_name']),
                'url': f"http://images.cocodataset.org/val2017/{img['file_name']}",
                'dest': os.path.join(coco_root, 'val2017', img['file_name']),
                'type': 'val'
            }
            for img in val_data['images']
        ]
        all_files_to_process.extend(val_files)
    
    # Step 1: Check for missing files (fast check)
    print(f"Checking existence of {len(all_files_to_process)} files...")
    missing_files = check_file_existence_batch(all_files_to_process)
    print(f"Found {len(missing_files)} missing files.")
    
    # Step 2: Check existing files for corruption (more resource-intensive)
    existing_files = [f for f in all_files_to_process if os.path.exists(f['path'])]
    print(f"Validating integrity of {len(existing_files)} existing files...")
    
    # Process in batches to manage memory
    batch_size = 10000
    all_corrupt_files = []
    
    for i in range(0, len(existing_files), batch_size):
        batch = existing_files[i:i + batch_size]
        print(f"Validating batch {i//batch_size + 1}/{(len(existing_files) + batch_size - 1)//batch_size}...")
        
        corrupt_files = await validate_images_batch_async(batch)
        all_corrupt_files.extend(corrupt_files)
        print(f"Found {len(corrupt_files)} corrupted files in this batch.")
    
    print(f"Total files to redownload: {len(missing_files) + len(all_corrupt_files)}")
    
    # Step 3: Download all missing and corrupt files
    all_files_to_download = missing_files + all_corrupt_files
    if all_files_to_download:
        # Group by file type for better reporting
        train_count = len([f for f in all_files_to_download if f['type'] == 'train'])
        val_count = len([f for f in all_files_to_download if f['type'] == 'val'])
        
        print(f"Downloading {len(all_files_to_download)} files:")
        print(f"  - {train_count} train images")
        print(f"  - {val_count} val images")
        
        # Process in reasonable batches
        download_batch_size = 1000
        url_dest_pairs = [(f['url'], f['dest']) for f in all_files_to_download]
        
        for i in range(0, len(url_dest_pairs), download_batch_size):
            batch = url_dest_pairs[i:i + download_batch_size]
            batch_num = i//download_batch_size + 1
            total_batches = (len(url_dest_pairs) + download_batch_size - 1)//download_batch_size
            
            print(f"Processing download batch {batch_num}/{total_batches}...")
            successes, failures = await download_multiple_files_async(batch)
            
            print(f"Successfully downloaded {len(successes)} files in batch {batch_num}.")
            if failures:
                print(f"Failed to download {len(failures)} files in batch {batch_num}.")
                if len(failures) <= 10:
                    for failure in failures:
                        if isinstance(failure, Exception):
                            print(f"  - Exception: {str(failure)}")
                        else:
                            print(f"  - {failure[1]}")
                else:
                    # Show just a summary if there are many failures
                    print(f"  - First 5 failures:")
                    for failure in failures[:5]:
                        if isinstance(failure, Exception):
                            print(f"    * Exception: {str(failure)}")
                        else:
                            print(f"    * {failure[1]}")
                    print(f"  ... and {len(failures) - 5} more")
    else:
        print("No missing or corrupted files found. Dataset integrity verified!")

    print("Dataset integrity check and repair completed!")

# Run the async main function
def check_and_download_coco(coco_root):
    asyncio.run(check_and_download_coco_async(coco_root))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Detection Dataset Downloader', add_help=True)

    parser.add_argument('--data-path', default='data\coco', help='dataset')
    args = parser.parse_args()
    
    check_and_download_coco(args.data_path)
